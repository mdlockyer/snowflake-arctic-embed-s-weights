import ArcticEmbedWeights
import Foundation

func main() -> Int32 {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 2 else {
        FileHandle.standardError.write(Data("usage: arctic-embed-convert <model.safetensors> <out.ae8>\n".utf8))
        return 2
    }
    let input = args[0]
    let output = args[1]
    let t0 = Double(DispatchTime.now().uptimeNanoseconds) * 1e-9

    let raw: QuantizedModelRaw
    do {
        raw = try QuantizedModelRaw.fromSafetensorsFile(input)
    } catch {
        FileHandle.standardError.write(Data("error: reading \(input): \(error)\n".utf8))
        return 1
    }

    do {
        try raw.saveAe8(output)
    } catch {
        FileHandle.standardError.write(Data("error: writing \(output): \(error)\n".utf8))
        return 1
    }

    let elapsedMs = (Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 - t0) * 1e3
    let bytes = (try? FileManager.default.attributesOfItem(atPath: output)[.size] as? UInt64) ?? 0
    print(String(format: "wrote %@: %.1f MB in %.0f ms", output, Double(bytes) / 1e6, elapsedMs))
    return 0
}

exit(main())
