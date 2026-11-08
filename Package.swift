// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "snowflake-arctic-embed-s-weights",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "ArcticEmbedWeights",
            targets: ["ArcticEmbedWeights"]
        ),
        .executable(
            name: "arctic-embed-convert",
            targets: ["arctic-embed-convert"]
        ),
    ],
    targets: [
        .target(
            name: "ArcticEmbedWeights",
            path: "Sources/ArcticEmbedWeights"
        ),
        .executableTarget(
            name: "arctic-embed-convert",
            dependencies: ["ArcticEmbedWeights"],
            path: "Sources/arctic-embed-convert"
        ),
        .testTarget(
            name: "ArcticEmbedWeightsTests",
            dependencies: ["ArcticEmbedWeights"],
            path: "Tests/ArcticEmbedWeightsTests"
        ),
    ]
)
