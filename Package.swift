// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "LyraCore",
  platforms: [.macOS("26.2")],
  products: [.library(name: "LyraCore", targets: ["LyraCore"])],
  targets: [
    // Built from the same native sources/MLX archives as the CLI by
    // `cmake --build build --target lyra_swift_package`.
    .binaryTarget(
      name: "CLyraCore",
      path: "native/swift/Artifacts/CLyraCore.xcframework"
    ),
    .systemLibrary(
      name: "CSndFile",
      path: "native/swift/CSndFile",
      pkgConfig: "sndfile",
      providers: [.brew(["libsndfile"])]
    ),
    .target(
      name: "LyraCore",
      dependencies: ["CLyraCore", "CSndFile"],
      path: "native/swift/LyraCore",
      sources: ["Engine.swift", "NativeBuild.swift"],
      resources: [.copy("Resources")],
      linkerSettings: [
        .linkedLibrary("c++"),
        .linkedLibrary("curl"),
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("MetalPerformanceShadersGraph"),
        .linkedFramework("Accelerate"),
        .linkedFramework("IOKit"),
      ]
    ),
  ]
)
