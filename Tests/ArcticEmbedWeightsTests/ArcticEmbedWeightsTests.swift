import ArcticEmbedWeights
import CryptoKit
import Foundation
import XCTest

final class ArcticEmbedWeightsTests: XCTestCase {
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testDetMathMatchesLibmClosely() {
        var k = -1000
        while k <= 1000 {
            let x = Double(k) * 0.01
            let a = DetMath.expF64(x)
            let b = Foundation.exp(x)
            let rel = abs(a - b) / b
            XCTAssertLessThan(rel, 1e-14, "exp(\(x)): det=\(a) std=\(b) rel=\(rel)")
            k += 7
        }
        for i in 1...1000 {
            let x = Double(i) * 0.0037
            let a = DetMath.lnF64(x)
            let b = Foundation.log(x)
            let err = abs(a - b)
            XCTAssertLessThan(err, 1e-14, "ln(\(x)): det=\(a) std=\(b) err=\(err)")
        }
        var j = -500
        while j <= 500 {
            let x = Float(j) * 0.01
            let d = abs(DetMath.gelu(x) - stdGelu(x))
            XCTAssertLessThan(d, 5e-7, "gelu(\(x)): diff=\(d)")
            j += 3
        }
    }

    private func stdGelu(_ x: Float) -> Float {
        let z = abs(x) * Float(1.0 / 2.0.squareRoot())
        let t: Float = 1.0 / (1.0 + 0.3275911 * z)
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        let erfMag = 1.0 - poly * Foundation.exp(-z * z)
        return 0.5 * x * (1.0 + Float(signOf: x, magnitudeOf: erfMag))
    }

    func testSafetensorsConversionDeterminismAndRoundtrip() throws {
        let stPath: URL
        if let dir = ProcessInfo.processInfo.environment["ARCTIC_EMBED_SOURCE_WEIGHTS"] {
            stPath = URL(fileURLWithPath: dir).appendingPathComponent("model.safetensors")
        } else {
            stPath = Self.repoRoot.appendingPathComponent("weights/model.safetensors")
        }
        guard FileManager.default.fileExists(atPath: stPath.path) else {
            print("\(stPath.path) not present; skipping conversion determinism check (run scripts/download_weights.sh to enable)")
            return
        }

        let converted = try QuantizedModelRaw.fromSafetensorsFile(stPath)
        let ae8Bytes = converted.toAe8Data()

        let digest = SHA256.hash(data: ae8Bytes)
        let shaHex = digest.map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(
            shaHex,
            EXPECTED_AE8_SHA256,
            "Converted model.ae8 SHA-256 did not match expected deterministic hash"
        )

        let roundtripped = try QuantizedModelRaw.loadAe8Data(ae8Bytes)
        XCTAssertEqual(roundtripped, converted)
        XCTAssertEqual(roundtripped.toAe8Data(), ae8Bytes)
    }
}
