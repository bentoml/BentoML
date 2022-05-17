target "_supported_platforms" {
	platforms = ["linux/amd64", "linux/arm64/v8", "linux/arm/v7", "linux/arm/v6", "linux/ppc64le", "linux/s390x", "linux/riscv64"]
}

target "platforms" {
	inherits = ["_supported_platforms"]
	dockerfile = "scripts/docker/platform.Dockerfile"
	no-cache = true
}
