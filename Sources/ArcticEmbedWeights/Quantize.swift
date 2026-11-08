import Foundation

public let HIDDEN_SIZE: Int = 384
public let INTERMEDIATE_SIZE: Int = 1536
public let NUM_HEADS: Int = 12
public let HEAD_DIM: Int = 32
public let NUM_LAYERS: Int = 12
public let VOCAB_SIZE: Int = 30522
public let MAX_POSITIONS: Int = 512
public let TYPE_VOCAB_SIZE: Int = 2
public let LAYER_NORM_EPS: Float = 1e-12
public let SMOOTHQUANT_ALPHA: Float = 0.40

public let AE8_MAGIC: [UInt8] = [0x41, 0x52, 0x43, 0x54, 0x49, 0x43, 0x38, 0x01] // "ARCTIC8\x01"
public let EXPECTED_AE8_SHA256 = "d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1"

public enum ArcticEmbedWeightsError: Error, CustomStringConvertible {
    case invalidMagic
    case unexpectedEOF
    case invalidUTF8Header
    case tensorNotFound(String)
    case ioError(String)

    public var description: String {
        switch self {
        case .invalidMagic:
            return "invalid .ae8 magic header"
        case .unexpectedEOF:
            return "unexpected end of file while reading model data"
        case .invalidUTF8Header:
            return "safetensors header is not valid UTF-8"
        case .tensorNotFound(let name):
            return "tensor '\(name)' not found in safetensors"
        case .ioError(let msg):
            return msg
        }
    }
}

public struct QuantizedLinearRaw: Sendable, Equatable {
    public var nOut: Int
    public var kIn: Int
    public var wQ: [Int8]
    public var wScale: [Float]
    public var bias: [Float]
    public var invSmooth: [Float]

    public init(
        nOut: Int,
        kIn: Int,
        wQ: [Int8],
        wScale: [Float],
        bias: [Float],
        invSmooth: [Float]
    ) {
        self.nOut = nOut
        self.kIn = kIn
        self.wQ = wQ
        self.wScale = wScale
        self.bias = bias
        self.invSmooth = invSmooth
    }
}

public struct QuantizedLayerRaw: Sendable, Equatable {
    public var qkv: QuantizedLinearRaw
    /// Separate Q (`[384, 384]`) and KV (`[768, 384]`) for Layer 11 CLS-only pruning.
    public var qOnly: QuantizedLinearRaw?
    public var kvOnly: QuantizedLinearRaw?
    public var attnOut: QuantizedLinearRaw
    public var ln1Weight: [Float]
    public var ln1Bias: [Float]
    public var ffnInter: QuantizedLinearRaw
    public var ffnOut: QuantizedLinearRaw
    public var ln2Weight: [Float]
    public var ln2Bias: [Float]
}

public struct QuantizedModelRaw: Sendable, Equatable {
    public var wordEmbQ: [Int8]
    public var wordEmbScale: [Float]
    public var posEmbQ: [Int8]
    public var posEmbScale: [Float]
    public var typeEmbQ: [Int8]
    public var typeEmbScale: [Float]
    public var embLnWeight: [Float]
    public var embLnBias: [Float]
    public var layers: [QuantizedLayerRaw]

    /// Loads FP32 weights from `model.safetensors`, runs deterministic SmoothQuant calibration
    /// (`alpha = 0.40`), and quantizes all weights & embedding tables to symmetric Int8.
    public static func fromSafetensorsFile(_ path: String) throws -> QuantizedModelRaw {
        try fromSafetensorsFile(URL(fileURLWithPath: path))
    }

    public static func fromSafetensorsFile(_ url: URL) throws -> QuantizedModelRaw {
        let data = try Data(contentsOf: url)
        return try data.withUnsafeBytes { rawBuf in
            try fromSafetensorsBytes(rawBuf.bindMemory(to: UInt8.self))
        }
    }

    public static func fromSafetensorsBytes(_ bytes: UnsafeBufferPointer<UInt8>) throws -> QuantizedModelRaw {
        let st = try SafeTensorsView.parse(bytes)

        let wordEmb = try st.getF32("embeddings.word_embeddings.weight")
        let posEmb = try st.getF32("embeddings.position_embeddings.weight")
        let typeEmb = try st.getF32("embeddings.token_type_embeddings.weight")
        let embLnWeight = try st.getF32("embeddings.LayerNorm.weight")
        let embLnBias = try st.getF32("embeddings.LayerNorm.bias")

        let (wordEmbQ, wordEmbScale) = quantizeRowsI8(wordEmb, rows: VOCAB_SIZE, cols: HIDDEN_SIZE)
        let (posEmbQ, posEmbScale) = quantizeRowsI8(posEmb, rows: MAX_POSITIONS, cols: HIDDEN_SIZE)
        let (typeEmbQ, typeEmbScale) = quantizeRowsI8(typeEmb, rows: TYPE_VOCAB_SIZE, cols: HIDDEN_SIZE)

        // Deterministic calibration sequence `[CLS, 2000..2030, SEP]` (length 32)
        var calibSeq: [Int] = []
        calibSeq.reserveCapacity(32)
        calibSeq.append(101)
        for id in 2000..<2030 {
            calibSeq.append(id)
        }
        calibSeq.append(102)
        let seqLen = calibSeq.count

        var x = [Float](repeating: 0.0, count: seqLen * HIDDEN_SIZE)
        for (t, tokId) in calibSeq.enumerated() {
            let rowOff = t * HIDDEN_SIZE
            let wOff = tokId * HIDDEN_SIZE
            let pOff = t * HIDDEN_SIZE
            for j in 0..<HIDDEN_SIZE {
                x[rowOff + j] = wordEmb[wOff + j] + posEmb[pOff + j] + typeEmb[j]
            }
            layernormF32Row(&x, rowOffset: rowOff, count: HIDDEN_SIZE, gamma: embLnWeight, beta: embLnBias)
        }

        var layers: [QuantizedLayerRaw] = []
        layers.reserveCapacity(NUM_LAYERS)

        for l in 0..<NUM_LAYERS {
            let prefix = "encoder.layer.\(l)."
            let wq = try st.getF32("\(prefix)attention.self.query.weight")
            let bq = try st.getF32("\(prefix)attention.self.query.bias")
            let wk = try st.getF32("\(prefix)attention.self.key.weight")
            let bk = try st.getF32("\(prefix)attention.self.key.bias")
            let wv = try st.getF32("\(prefix)attention.self.value.weight")
            let bv = try st.getF32("\(prefix)attention.self.value.bias")

            var wQkv: [Float] = []
            wQkv.reserveCapacity(3 * HIDDEN_SIZE * HIDDEN_SIZE)
            wQkv.append(contentsOf: wq)
            wQkv.append(contentsOf: wk)
            wQkv.append(contentsOf: wv)

            var bQkv: [Float] = []
            bQkv.reserveCapacity(3 * HIDDEN_SIZE)
            bQkv.append(contentsOf: bq)
            bQkv.append(contentsOf: bk)
            bQkv.append(contentsOf: bv)

            let sQkv = computeSmoothScales(
                x: x,
                m: seqLen,
                k: HIDDEN_SIZE,
                w: wQkv,
                n: 3 * HIDDEN_SIZE,
                alpha: SMOOTHQUANT_ALPHA
            )

            let qkvQuant: QuantizedLinearRaw
            let qOnly: QuantizedLinearRaw?
            let kvOnly: QuantizedLinearRaw?

            if l == NUM_LAYERS - 1 {
                var wKv: [Float] = []
                wKv.reserveCapacity(2 * HIDDEN_SIZE * HIDDEN_SIZE)
                wKv.append(contentsOf: wk)
                wKv.append(contentsOf: wv)

                var bKv: [Float] = []
                bKv.reserveCapacity(2 * HIDDEN_SIZE)
                bKv.append(contentsOf: bk)
                bKv.append(contentsOf: bv)

                let invSmooth = sQkv.map { 1.0 / $0 }
                qkvQuant = QuantizedLinearRaw(
                    nOut: 0,
                    kIn: HIDDEN_SIZE,
                    wQ: [],
                    wScale: [],
                    bias: [],
                    invSmooth: invSmooth
                )
                qOnly = quantizeLinearWithSmooth(
                    nOut: HIDDEN_SIZE,
                    kIn: HIDDEN_SIZE,
                    w: wq,
                    bias: bq,
                    smooth: sQkv
                )
                kvOnly = quantizeLinearWithSmooth(
                    nOut: 2 * HIDDEN_SIZE,
                    kIn: HIDDEN_SIZE,
                    w: wKv,
                    bias: bKv,
                    smooth: sQkv
                )
            } else {
                qkvQuant = quantizeLinearWithSmooth(
                    nOut: 3 * HIDDEN_SIZE,
                    kIn: HIDDEN_SIZE,
                    w: wQkv,
                    bias: bQkv,
                    smooth: sQkv
                )
                qOnly = nil
                kvOnly = nil
            }

            // Propagate FP32 reference activation to calibrate attention output
            let q = matmulFp32(a: x, m: seqLen, k: HIDDEN_SIZE, w: wq, n: HIDDEN_SIZE, bias: bq)
            let k = matmulFp32(a: x, m: seqLen, k: HIDDEN_SIZE, w: wk, n: HIDDEN_SIZE, bias: bk)
            let v = matmulFp32(a: x, m: seqLen, k: HIDDEN_SIZE, w: wv, n: HIDDEN_SIZE, bias: bv)
            let ctx = attentionFp32(q: q, k: k, v: v, seqLen: seqLen)

            let wAo = try st.getF32("\(prefix)attention.output.dense.weight")
            let bAo = try st.getF32("\(prefix)attention.output.dense.bias")
            let sAo = computeSmoothScales(
                x: ctx,
                m: seqLen,
                k: HIDDEN_SIZE,
                w: wAo,
                n: HIDDEN_SIZE,
                alpha: SMOOTHQUANT_ALPHA
            )
            let attnOutQuant = quantizeLinearWithSmooth(
                nOut: HIDDEN_SIZE,
                kIn: HIDDEN_SIZE,
                w: wAo,
                bias: bAo,
                smooth: sAo
            )

            let attnOut = matmulFp32(a: ctx, m: seqLen, k: HIDDEN_SIZE, w: wAo, n: HIDDEN_SIZE, bias: bAo)
            let ln1Weight = try st.getF32("\(prefix)attention.output.LayerNorm.weight")
            let ln1Bias = try st.getF32("\(prefix)attention.output.LayerNorm.bias")
            for t in 0..<seqLen {
                let rowOff = t * HIDDEN_SIZE
                for j in 0..<HIDDEN_SIZE {
                    x[rowOff + j] += attnOut[rowOff + j]
                }
                layernormF32Row(&x, rowOffset: rowOff, count: HIDDEN_SIZE, gamma: ln1Weight, beta: ln1Bias)
            }

            let wInter = try st.getF32("\(prefix)intermediate.dense.weight")
            let bInter = try st.getF32("\(prefix)intermediate.dense.bias")
            let sInter = computeSmoothScales(
                x: x,
                m: seqLen,
                k: HIDDEN_SIZE,
                w: wInter,
                n: INTERMEDIATE_SIZE,
                alpha: SMOOTHQUANT_ALPHA
            )
            let ffnInterQuant = quantizeLinearWithSmooth(
                nOut: INTERMEDIATE_SIZE,
                kIn: HIDDEN_SIZE,
                w: wInter,
                bias: bInter,
                smooth: sInter
            )

            var inter = matmulFp32(
                a: x,
                m: seqLen,
                k: HIDDEN_SIZE,
                w: wInter,
                n: INTERMEDIATE_SIZE,
                bias: bInter
            )
            for idx in 0..<inter.count {
                inter[idx] = DetMath.gelu(inter[idx])
            }

            let wOut = try st.getF32("\(prefix)output.dense.weight")
            let bOut = try st.getF32("\(prefix)output.dense.bias")
            let sOut = computeSmoothScales(
                x: inter,
                m: seqLen,
                k: INTERMEDIATE_SIZE,
                w: wOut,
                n: HIDDEN_SIZE,
                alpha: SMOOTHQUANT_ALPHA
            )
            let ffnOutQuant = quantizeLinearWithSmooth(
                nOut: HIDDEN_SIZE,
                kIn: INTERMEDIATE_SIZE,
                w: wOut,
                bias: bOut,
                smooth: sOut
            )

            let ffnOut = matmulFp32(
                a: inter,
                m: seqLen,
                k: INTERMEDIATE_SIZE,
                w: wOut,
                n: HIDDEN_SIZE,
                bias: bOut
            )
            let ln2Weight = try st.getF32("\(prefix)output.LayerNorm.weight")
            let ln2Bias = try st.getF32("\(prefix)output.LayerNorm.bias")
            for t in 0..<seqLen {
                let rowOff = t * HIDDEN_SIZE
                for j in 0..<HIDDEN_SIZE {
                    x[rowOff + j] += ffnOut[rowOff + j]
                }
                layernormF32Row(&x, rowOffset: rowOff, count: HIDDEN_SIZE, gamma: ln2Weight, beta: ln2Bias)
            }

            layers.append(
                QuantizedLayerRaw(
                    qkv: qkvQuant,
                    qOnly: qOnly,
                    kvOnly: kvOnly,
                    attnOut: attnOutQuant,
                    ln1Weight: ln1Weight,
                    ln1Bias: ln1Bias,
                    ffnInter: ffnInterQuant,
                    ffnOut: ffnOutQuant,
                    ln2Weight: ln2Weight,
                    ln2Bias: ln2Bias
                )
            )
        }

        return QuantizedModelRaw(
            wordEmbQ: wordEmbQ,
            wordEmbScale: wordEmbScale,
            posEmbQ: posEmbQ,
            posEmbScale: posEmbScale,
            typeEmbQ: typeEmbQ,
            typeEmbScale: typeEmbScale,
            embLnWeight: embLnWeight,
            embLnBias: embLnBias,
            layers: layers
        )
    }

    /// Saves the quantized model in portable `.ae8` binary format (~33.8 MB).
    public func saveAe8(_ path: String) throws {
        try saveAe8(URL(fileURLWithPath: path))
    }

    public func saveAe8(_ url: URL) throws {
        try toAe8Data().write(to: url)
    }

    public func toAe8Data() -> Data {
        var out = Data()
        out.reserveCapacity(34_000_000)
        out.append(contentsOf: AE8_MAGIC)
        writeI8Slice(&out, wordEmbQ)
        writeF32Slice(&out, wordEmbScale)
        writeI8Slice(&out, posEmbQ)
        writeF32Slice(&out, posEmbScale)
        writeI8Slice(&out, typeEmbQ)
        writeF32Slice(&out, typeEmbScale)
        writeF32Slice(&out, embLnWeight)
        writeF32Slice(&out, embLnBias)

        for l in layers {
            writeLinear(&out, l.qkv)
            if let q = l.qOnly, let kv = l.kvOnly {
                out.append(1)
                writeLinear(&out, q)
                writeLinear(&out, kv)
            } else {
                out.append(0)
            }
            writeLinear(&out, l.attnOut)
            writeF32Slice(&out, l.ln1Weight)
            writeF32Slice(&out, l.ln1Bias)
            writeLinear(&out, l.ffnInter)
            writeLinear(&out, l.ffnOut)
            writeF32Slice(&out, l.ln2Weight)
            writeF32Slice(&out, l.ln2Bias)
        }
        return out
    }

    /// Loads the quantized model from portable `.ae8` binary format.
    public static func loadAe8(_ path: String) throws -> QuantizedModelRaw {
        try loadAe8(URL(fileURLWithPath: path))
    }

    public static func loadAe8(_ url: URL) throws -> QuantizedModelRaw {
        let data = try Data(contentsOf: url)
        return try loadAe8Data(data)
    }

    public static func loadAe8Data(_ data: Data) throws -> QuantizedModelRaw {
        try data.withUnsafeBytes { rawBuf in
            let bytes = rawBuf.bindMemory(to: UInt8.self)
            guard let base = bytes.baseAddress, bytes.count >= 8 else {
                throw ArcticEmbedWeightsError.invalidMagic
            }
            for i in 0..<8 {
                if base[i] != AE8_MAGIC[i] {
                    throw ArcticEmbedWeightsError.invalidMagic
                }
            }
            var cursor = 8
            let total = bytes.count

            let wordEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let wordEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let posEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let posEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let typeEmbQ = try readI8Vec(base, cursor: &cursor, total: total)
            let typeEmbScale = try readF32Vec(base, cursor: &cursor, total: total)
            let embLnWeight = try readF32Vec(base, cursor: &cursor, total: total)
            let embLnBias = try readF32Vec(base, cursor: &cursor, total: total)

            var layers: [QuantizedLayerRaw] = []
            layers.reserveCapacity(NUM_LAYERS)

            for _ in 0..<NUM_LAYERS {
                let qkv = try readLinear(base, cursor: &cursor, total: total)
                guard cursor < total else { throw ArcticEmbedWeightsError.unexpectedEOF }
                let hasSplit = base[cursor]
                cursor += 1
                let qOnly: QuantizedLinearRaw?
                let kvOnly: QuantizedLinearRaw?
                if hasSplit == 1 {
                    qOnly = try readLinear(base, cursor: &cursor, total: total)
                    kvOnly = try readLinear(base, cursor: &cursor, total: total)
                } else {
                    qOnly = nil
                    kvOnly = nil
                }
                let attnOut = try readLinear(base, cursor: &cursor, total: total)
                let ln1Weight = try readF32Vec(base, cursor: &cursor, total: total)
                let ln1Bias = try readF32Vec(base, cursor: &cursor, total: total)
                let ffnInter = try readLinear(base, cursor: &cursor, total: total)
                let ffnOut = try readLinear(base, cursor: &cursor, total: total)
                let ln2Weight = try readF32Vec(base, cursor: &cursor, total: total)
                let ln2Bias = try readF32Vec(base, cursor: &cursor, total: total)

                layers.append(
                    QuantizedLayerRaw(
                        qkv: qkv,
                        qOnly: qOnly,
                        kvOnly: kvOnly,
                        attnOut: attnOut,
                        ln1Weight: ln1Weight,
                        ln1Bias: ln1Bias,
                        ffnInter: ffnInter,
                        ffnOut: ffnOut,
                        ln2Weight: ln2Weight,
                        ln2Bias: ln2Bias
                    )
                )
            }

            return QuantizedModelRaw(
                wordEmbQ: wordEmbQ,
                wordEmbScale: wordEmbScale,
                posEmbQ: posEmbQ,
                posEmbScale: posEmbScale,
                typeEmbQ: typeEmbQ,
                typeEmbScale: typeEmbScale,
                embLnWeight: embLnWeight,
                embLnBias: embLnBias,
                layers: layers
            )
        }
    }
}

// MARK: - Binary Serialization Helpers

@inline(__always)
private func writeU64LE(_ out: inout Data, _ v: UInt64) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
}

private func writeI8Slice(_ out: inout Data, _ s: [Int8]) {
    writeU64LE(&out, UInt64(s.count))
    s.withUnsafeBytes { out.append(contentsOf: $0) }
}

private func writeF32Slice(_ out: inout Data, _ s: [Float]) {
    writeU64LE(&out, UInt64(s.count))
    s.withUnsafeBytes { out.append(contentsOf: $0) }
}

private func writeLinear(_ out: inout Data, _ lin: QuantizedLinearRaw) {
    writeU64LE(&out, UInt64(lin.nOut))
    writeU64LE(&out, UInt64(lin.kIn))
    writeI8Slice(&out, lin.wQ)
    writeF32Slice(&out, lin.wScale)
    writeF32Slice(&out, lin.bias)
    writeF32Slice(&out, lin.invSmooth)
}

@inline(__always)
private func readU64LE(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> Int {
    guard cursor + 8 <= total else { throw ArcticEmbedWeightsError.unexpectedEOF }
    let v = UnsafeRawPointer(base.advanced(by: cursor)).loadUnaligned(as: UInt64.self)
    cursor += 8
    return Int(UInt64(littleEndian: v))
}

private func readI8Vec(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> [Int8] {
    let len = try readU64LE(base, cursor: &cursor, total: total)
    guard cursor + len <= total else { throw ArcticEmbedWeightsError.unexpectedEOF }
    var out = [Int8](repeating: 0, count: len)
    if len > 0 {
        out.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(from: UnsafeRawPointer(base.advanced(by: cursor)), byteCount: len)
        }
    }
    cursor += len
    return out
}

private func readF32Vec(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> [Float] {
    let len = try readU64LE(base, cursor: &cursor, total: total)
    let byteCount = len * 4
    guard cursor + byteCount <= total else { throw ArcticEmbedWeightsError.unexpectedEOF }
    var out = [Float](repeating: 0.0, count: len)
    if byteCount > 0 {
        out.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(from: UnsafeRawPointer(base.advanced(by: cursor)), byteCount: byteCount)
        }
    }
    cursor += byteCount
    return out
}

private func readLinear(_ base: UnsafePointer<UInt8>, cursor: inout Int, total: Int) throws -> QuantizedLinearRaw {
    let nOut = try readU64LE(base, cursor: &cursor, total: total)
    let kIn = try readU64LE(base, cursor: &cursor, total: total)
    let wQ = try readI8Vec(base, cursor: &cursor, total: total)
    let wScale = try readF32Vec(base, cursor: &cursor, total: total)
    let bias = try readF32Vec(base, cursor: &cursor, total: total)
    let invSmooth = try readF32Vec(base, cursor: &cursor, total: total)
    return QuantizedLinearRaw(
        nOut: nOut,
        kIn: kIn,
        wQ: wQ,
        wScale: wScale,
        bias: bias,
        invSmooth: invSmooth
    )
}

// MARK: - SmoothQuant Calibration & SafeTensors Parser

private func quantizeRowsI8(_ data: [Float], rows: Int, cols: Int) -> ([Int8], [Float]) {
    var q = [Int8](repeating: 0, count: rows * cols)
    var scales = [Float](repeating: 0.0, count: rows)
    for r in 0..<rows {
        let rowOff = r * cols
        var maxAbs: Float = 1e-8
        for c in 0..<cols {
            let a = abs(data[rowOff + c])
            if a > maxAbs {
                maxAbs = a
            }
        }
        let scale = maxAbs / 127.0
        let invScale = 127.0 / maxAbs
        scales[r] = scale
        for c in 0..<cols {
            let v = (data[rowOff + c] * invScale).rounded(.toNearestOrAwayFromZero)
            let clamped = min(max(v, -127.0), 127.0)
            q[rowOff + c] = Int8(clamped)
        }
    }
    return (q, scales)
}

private func computeSmoothScales(
    x: [Float],
    m: Int,
    k: Int,
    w: [Float],
    n: Int,
    alpha: Float
) -> [Float] {
    var actMax = [Float](repeating: 1e-5, count: k)
    for r in 0..<m {
        let rowOff = r * k
        for j in 0..<k {
            let a = abs(x[rowOff + j])
            if a > actMax[j] {
                actMax[j] = a
            }
        }
    }
    var wMax = [Float](repeating: 1e-5, count: k)
    for rowIdx in 0..<n {
        let rowOff = rowIdx * k
        for j in 0..<k {
            let a = abs(w[rowOff + j])
            if a > wMax[j] {
                wMax[j] = a
            }
        }
    }
    var s = [Float](repeating: 1.0, count: k)
    for j in 0..<k {
        s[j] = DetMath.pow(actMax[j], alpha) / DetMath.pow(wMax[j], 1.0 - alpha)
    }
    return s
}

private func quantizeLinearWithSmooth(
    nOut: Int,
    kIn: Int,
    w: [Float],
    bias: [Float],
    smooth: [Float]
) -> QuantizedLinearRaw {
    var wSmoothed = [Float](repeating: 0.0, count: nOut * kIn)
    for r in 0..<nOut {
        let off = r * kIn
        for j in 0..<kIn {
            wSmoothed[off + j] = w[off + j] * smooth[j]
        }
    }
    let (wQ, wScale) = quantizeRowsI8(wSmoothed, rows: nOut, cols: kIn)
    let invSmooth = smooth.map { 1.0 / $0 }
    return QuantizedLinearRaw(
        nOut: nOut,
        kIn: kIn,
        wQ: wQ,
        wScale: wScale,
        bias: bias,
        invSmooth: invSmooth
    )
}

private func layernormF32Row(
    _ x: inout [Float],
    rowOffset: Int,
    count: Int,
    gamma: [Float],
    beta: [Float]
) {
    var sum: Float = 0.0
    for j in 0..<count {
        sum += x[rowOffset + j]
    }
    let mean = sum / Float(count)
    var varSum: Float = 0.0
    for j in 0..<count {
        let d = x[rowOffset + j] - mean
        varSum += d * d
    }
    let variance = varSum / Float(count)
    let rstd = 1.0 / (variance + LAYER_NORM_EPS).squareRoot()
    for j in 0..<count {
        x[rowOffset + j] = (x[rowOffset + j] - mean) * rstd * gamma[j] + beta[j]
    }
}

private func matmulFp32(
    a: [Float],
    m: Int,
    k: Int,
    w: [Float],
    n: Int,
    bias: [Float]
) -> [Float] {
    var out = [Float](repeating: 0.0, count: m * n)
    a.withUnsafeBufferPointer { aBuf in
        w.withUnsafeBufferPointer { wBuf in
            bias.withUnsafeBufferPointer { bBuf in
                out.withUnsafeMutableBufferPointer { outBuf in
                    let ap = aBuf.baseAddress!
                    let wp = wBuf.baseAddress!
                    let bp = bBuf.baseAddress!
                    let op = outBuf.baseAddress!
                    for r in 0..<m {
                        let aRow = ap.advanced(by: r * k)
                        let outRow = op.advanced(by: r * n)
                        for col in 0..<n {
                            let wRow = wp.advanced(by: col * k)
                            var sum = bp[col]
                            for j in 0..<k {
                                sum += aRow[j] * wRow[j]
                            }
                            outRow[col] = sum
                        }
                    }
                }
            }
        }
    }
    return out
}

private func attentionFp32(q: [Float], k: [Float], v: [Float], seqLen: Int) -> [Float] {
    var ctx = [Float](repeating: 0.0, count: seqLen * HIDDEN_SIZE)
    let scale = 1.0 / Float(HEAD_DIM).squareRoot()
    for h in 0..<NUM_HEADS {
        let hOff = h * HEAD_DIM
        for tq in 0..<seqLen {
            let qOff = tq * HIDDEN_SIZE + hOff
            var scores = [Float](repeating: 0.0, count: seqLen)
            var maxS = -Float.infinity
            for tk in 0..<seqLen {
                let kOff = tk * HIDDEN_SIZE + hOff
                var dot: Float = 0.0
                for d in 0..<HEAD_DIM {
                    dot += q[qOff + d] * k[kOff + d]
                }
                let s = dot * scale
                scores[tk] = s
                if s > maxS {
                    maxS = s
                }
            }
            var sumExp: Float = 0.0
            for tk in 0..<seqLen {
                let e = DetMath.exp(scores[tk] - maxS)
                scores[tk] = e
                sumExp += e
            }
            for tk in 0..<seqLen {
                scores[tk] /= sumExp
            }
            let outOff = tq * HIDDEN_SIZE + hOff
            for tk in 0..<seqLen {
                let p = scores[tk]
                let vOff = tk * HIDDEN_SIZE + hOff
                for d in 0..<HEAD_DIM {
                    ctx[outOff + d] += p * v[vOff + d]
                }
            }
        }
    }
    return ctx
}

private struct SafeTensorsView {
    let base: UnsafePointer<UInt8>
    let dataOffset: Int
    let totalBytes: Int
    let offsets: [String: (Int, Int)]

    static func parse(_ bytes: UnsafeBufferPointer<UInt8>) throws -> SafeTensorsView {
        guard let base = bytes.baseAddress, bytes.count >= 8 else {
            throw ArcticEmbedWeightsError.unexpectedEOF
        }
        let headerLenU64 = UnsafeRawPointer(base).loadUnaligned(as: UInt64.self)
        let headerLen = Int(UInt64(littleEndian: headerLenU64))
        guard bytes.count >= 8 + headerLen else {
            throw ArcticEmbedWeightsError.unexpectedEOF
        }
        let headerSlice = UnsafeBufferPointer(start: base.advanced(by: 8), count: headerLen)
        guard let headerStr = String(bytes: headerSlice, encoding: .utf8) else {
            throw ArcticEmbedWeightsError.invalidUTF8Header
        }
        let offsets = parseSafetensorsHeader(headerStr)
        return SafeTensorsView(
            base: base,
            dataOffset: 8 + headerLen,
            totalBytes: bytes.count,
            offsets: offsets
        )
    }

    func getF32(_ name: String) throws -> [Float] {
        guard let (start, end) = offsets[name] else {
            throw ArcticEmbedWeightsError.tensorNotFound(name)
        }
        let byteLen = end - start
        let count = byteLen / 4
        var out = [Float](repeating: 0.0, count: count)
        if byteLen > 0 {
            out.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(
                    from: UnsafeRawPointer(base.advanced(by: dataOffset + start)),
                    byteCount: byteLen
                )
            }
        }
        return out
    }
}

private func parseSafetensorsHeader(_ s: String) -> [String: (Int, Int)] {
    var map: [String: (Int, Int)] = [:]
    let utf8 = Array(s.utf8)
    let n = utf8.count
    var pos = 0

    func findByte(_ b: UInt8, from start: Int, to end: Int) -> Int? {
        var i = start
        while i < end {
            if utf8[i] == b { return i }
            i += 1
        }
        return nil
    }

    while let q1Open = findByte(0x22, from: pos, to: n) {
        let q1 = q1Open + 1
        guard let q2 = findByte(0x22, from: q1, to: n) else { break }
        let key = String(decoding: utf8[q1..<q2], as: UTF8.self)
        pos = q2 + 1

        while pos < n && (utf8[pos] == 0x20 || utf8[pos] == 0x09 || utf8[pos] == 0x0A || utf8[pos] == 0x0D) {
            pos += 1
        }
        if pos >= n || utf8[pos] != 0x3A { // ':'
            continue
        }
        pos += 1
        while pos < n && (utf8[pos] == 0x20 || utf8[pos] == 0x09 || utf8[pos] == 0x0A || utf8[pos] == 0x0D) {
            pos += 1
        }
        if pos >= n || utf8[pos] != 0x7B { // '{'
            continue
        }
        let objStart = pos
        var depth = 0
        while pos < n {
            if utf8[pos] == 0x7B {
                depth += 1
            } else if utf8[pos] == 0x7D {
                depth -= 1
                if depth == 0 {
                    pos += 1
                    break
                }
            }
            pos += 1
        }
        if key == "__metadata__" {
            continue
        }
        let objStr = String(decoding: utf8[objStart..<pos], as: UTF8.self)
        if let range = objStr.range(of: "\"data_offsets\"") {
            let after = objStr[range.upperBound...]
            if let bOpen = after.firstIndex(of: "["),
               let bClose = after.firstIndex(of: "]"),
               bOpen < bClose
            {
                let nums = after[after.index(after: bOpen)..<bClose]
                let parts = nums.split(separator: ",")
                if parts.count >= 2,
                   let start = Int(parts[0].trimmingCharacters(in: .whitespacesAndNewlines)),
                   let end = Int(parts[1].trimmingCharacters(in: .whitespacesAndNewlines))
                {
                    map[key] = (start, end)
                }
            }
        }
    }
    return map
}

/// Platform-independent `exp`/`ln`/`pow`/GELU for the converter.
///
/// Uses only `+ - * /`, comparisons and bit manipulation in `Double` — all correctly rounded by
/// IEEE 754, and Swift never contracts them into FMAs — so the same checkpoint converts to a
/// byte-identical `model.ae8` everywhere.
public enum DetMath {
    private static let LN2_HI: Double = 6.931_471_803_691_238_164_90e-1
    private static let LN2_LO: Double = 1.908_214_929_270_587_700_02e-10
    private static let LN2: Double = 0.693147180559945309417232121458176568
    private static let FRAC_1_SQRT_2_F64: Double = 0.707106781186547524400844362104849039
    private static let FRAC_1_SQRT_2_F32: Float = 0.707106781186547524400844362104849039

    public static func expF64(_ x: Double) -> Double {
        if x.isNaN { return x }
        if x > 709.0 { return .infinity }
        if x < -745.0 { return 0.0 }
        let n = (x / LN2).rounded(.toNearestOrAwayFromZero)
        let r = (x - n * LN2_HI) - n * LN2_LO
        var p = 1.0 / factorial(13)
        for k in stride(from: 12, through: 0, by: -1) {
            p = p * r + 1.0 / factorial(UInt32(k))
        }
        return scaleByPow2(p, Int32(n))
    }

    @inline(__always)
    private static func factorial(_ k: UInt32) -> Double {
        if k == 0 { return 1.0 }
        var acc: Double = 1.0
        for i in 1...k {
            acc *= Double(i)
        }
        return acc
    }

    @inline(__always)
    private static func scaleByPow2(_ xIn: Double, _ nIn: Int32) -> Double {
        var x = xIn
        var n = nIn
        while n > 1000 {
            x *= Double(bitPattern: UInt64(1023 + 1000) << 52)
            n -= 1000
        }
        while n < -1000 {
            x *= Double(bitPattern: UInt64(1023 - 1000) << 52)
            n += 1000
        }
        return x * Double(bitPattern: UInt64(Int64(1023 + n)) << 52)
    }

    public static func lnF64(_ x: Double) -> Double {
        if x.isNaN || x < 0.0 { return .nan }
        if x == 0.0 { return -.infinity }
        if x.isInfinite { return x }
        var (m, e) = frexp(x)
        if m < FRAC_1_SQRT_2_F64 {
            m *= 2.0
            e -= 1
        }
        let s = (m - 1.0) / (m + 1.0)
        let s2 = s * s
        var p = 1.0 / 27.0
        var k: Int32 = 25
        while k >= 1 {
            p = p * s2 + 1.0 / Double(k)
            k -= 2
        }
        return 2.0 * s * p + Double(e) * LN2
    }

    @inline(__always)
    private static func frexp(_ xIn: Double) -> (Double, Int32) {
        var x = xIn
        var bias: Int32 = 0
        if x < Double.leastNormalMagnitude {
            x *= Double(bitPattern: (1023 + 54) << 52)
            bias = -54
        }
        let bits = x.bitPattern
        let e = Int32((bits >> 52) & 0x7ff) - 1022
        let m = Double(bitPattern: (bits & ~(0x7ff << 52)) | (1022 << 52))
        return (m, e + bias)
    }

    public static func exp(_ x: Float) -> Float {
        Float(expF64(Double(x)))
    }

    public static func pow(_ base: Float, _ e: Float) -> Float {
        if base == 0.0 {
            if e > 0.0 { return 0.0 }
            if e == 0.0 { return 1.0 }
            return .infinity
        }
        return Float(expF64(Double(e) * lnF64(Double(base))))
    }

    public static func gelu(_ x: Float) -> Float {
        let z = abs(x) * FRAC_1_SQRT_2_F32
        let t: Float = 1.0 / (1.0 + 0.3275911 * z)
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        let erfMag = 1.0 - poly * exp(-z * z)
        let erfVal = Float(signOf: x, magnitudeOf: erfMag)
        return 0.5 * x * (1.0 + erfVal)
    }
}
