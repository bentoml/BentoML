/* ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗ */
/* ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝ */
/* ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗ */
/* ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║ */
/* ██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║ */
/* ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ */

variable "NO_ARCH" {
    default = ""
}
function "TagWithArch" {
    params = [repo, tag, no_arch, arch]
    result = [no_arch =="1"?"${repo}:${tag}":"${repo}:${tag}-${arch}",]
}

/* ██╗  ██╗██╗  ██╗     ███╗   ███╗███████╗████████╗ █████╗ */
/* ╚██╗██╔╝╚██╗██╔╝     ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗ */
/*  ╚███╔╝  ╚███╔╝█████╗██╔████╔██║█████╗     ██║   ███████║ */
/*  ██╔██╗  ██╔██╗╚════╝██║╚██╔╝██║██╔══╝     ██║   ██╔══██║ */
/* ██╔╝ ██╗██╔╝ ██╗     ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║ */
/* ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ */

variable "XX_REPO" {
    default = "xx-local"
}
variable "XX_TAG"{
    default = "1.1.0"
}
target "_xx_meta"{
    tags = ["${XX_REPO}:${XX_TAG}"]
}
target "_xx-platforms" {
    platforms = ["linux/amd64", "linux/arm64", "linux/arm", "linux/arm/v6", "linux/ppc64le", "linux/s390x", "linux/386", "linux/riscv64"]
}
target "local-xx" {
    inherits = ["_xx_meta"]
	context = "vendors/xx/base"
	target = "base"
    tags = TagWithArch(XX_REPO, XX_TAG, "1", "")
}

target "local-xx-arm64" {
    inherits = ["local-xx"]
    platforms = ["linux/arm64/v8"]
    tags = TagWithArch(XX_REPO, XX_TAG, NO_ARCH, "arm64")
    output = ["type=docker"]
}

target "local-xx-amd64" {
    inherits = ["local-xx"]
    platforms = ["linux/amd64"]
    tags = TagWithArch(XX_REPO, XX_TAG, NO_ARCH, "amd64")
    output = ["type=docker"]
}
target "local-xx-s390x" {
    inherits = ["local-xx"]
    platforms = ["linux/s390x"]
    tags = TagWithArch(XX_REPO, XX_TAG, NO_ARCH, "s390x")
    output = ["type=docker"]
}
target "local-xx-ppc64le" {
    inherits = ["local-xx"]
    platforms = ["linux/ppc64le"]
    tags = TagWithArch(XX_REPO, XX_TAG, NO_ARCH, "ppc64le")
    output = ["type=docker"]
}

target "local-xx-all" {
    inherits = ["local-xx", "_xx-platforms"]
}

group "xx-default" {
    targets = ["local-xx-all"]
}

/* ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗       ███╗   ███╗███████╗████████╗ █████╗ */
/* ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗      ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗ */
/* ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝█████╗██╔████╔██║█████╗     ██║   ███████║ */
/* ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗╚════╝██║╚██╔╝██║██╔══╝     ██║   ██╔══██║ */
/* ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║      ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║ */
/* ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ */

variable "MANAGER_REPO" {
    default = "bentoml-docker"
}
variable "MANAGER_TAG"{
    default = "1.1.0"
}

group default {
   targets = ["manager-arm64", "manager-amd64"]
}

target "manager-arm64" {
    platforms = ["linux/arm64/v8"]
    dockerfile = "./hack/dev.Dockerfile"
    tags = TagWithArch(MANAGER_REPO, MANAGER_TAG, NO_ARCH, "arm64")
    output = ["type=docker"]
}

target "manager-amd64" {
    platforms = ["linux/amd64"]
    dockerfile = "./hack/dev.Dockerfile"
    tags = TagWithArch(MANAGER_REPO, MANAGER_TAG, NO_ARCH, "amd64")
    output = ["type=docker"]
}

/* ██████╗ ██╗      █████╗ ████████╗███████╗ ██████╗ ██████╗ ███╗   ███╗      ███╗   ███╗███████╗████████╗ █████╗ */
/* ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║      ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗ */
/* ██████╔╝██║     ███████║   ██║   █████╗  ██║   ██║██████╔╝██╔████╔██║█████╗██╔████╔██║█████╗     ██║   ███████║ */
/* ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║╚════╝██║╚██╔╝██║██╔══╝     ██║   ██╔══██║ */
/* ██║     ███████╗██║  ██║   ██║   ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║      ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║ */
/* ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝      ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ */

target "_test-platforms" {
    platforms = concat(["linux/amd64", "linux/386", "linux/arm64/v8", "linux/arm/v7","linux/arm/v6", "linux/ppc64le", "linux/s390x", "linux/riscv64","linux/mips64le"])
}
target "test-platforms" {
    inherits = ["_test-platforms"]
    dockerfile = "./hack/platform.Dockerfile"
}
