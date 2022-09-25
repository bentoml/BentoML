plugins {
    id("com.android.application") version "7.0.4" apply false // Older for IntelliJ Support
    id("com.google.protobuf") version "0.8.18" apply false
    kotlin("jvm") version "1.7.0" apply false
    id("org.jlleitschuh.gradle.ktlint") version "10.2.0"
    `java-library`
}

ext["grpcVersion"] = "1.48.0"
ext["grpcKotlinVersion"] = "1.6.0" // CURRENT_GRPC_KOTLIN_VERSION
ext["protobufVersion"] = "3.19.4"
ext["coroutinesVersion"] = "1.6.2"

allprojects {
    repositories {
        mavenLocal()
        mavenCentral()
        google()
    }

    apply(plugin = "org.jlleitschuh.gradle.ktlint")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(8))
    }
    sourceSets.getByName("main").resources.srcDir("src/main/proto")
}

dependencies {
    implementation(platform("org.jetbrains.kotlin:kotlin-bom"))

    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")

    implementation("com.google.guava:guava:30.1.1-jre")

    runtimeOnly("io.grpc:grpc-netty:${rootProject.ext["grpcVersion"]}")
    api(kotlin("stdlib-jdk8"))
    api("org.jetbrains.kolinx:kotlinx-coroutines-core:${rootProject.ext["coroutinesVersion"]}")
    api("io.grpc:grpc-stub:${rootProject.ext["grpcVersion"]}")
    api("io.grpc:grpc-protobuf:${rootProject.ext["grpcVersion"]}")
    api("com.google.protobuf:protobuf-java-util:${rootProject.ext["protobufVersion"]}")
    api("com.google.protobuf:protobuf-kotlin:${rootProject.ext["protobufVersion"]}")
    api("io.grpc:grpc-kotlin-stub:${rootProject.ext["grpcKotlinVersion"]}")
}

tasks.register<JavaExec>("BentoServiceClient") {
    dependsOn("classes")
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.client.BentoServiceClientKt")
}

val bentoServiceClientStartScripts = tasks.register<CreateStartScripts>("bentoServiceClientStartScripts") {
    mainClass.set("com.client.BentoServiceClientKt")
    applicationName = "bento-service-client"
    outputDir = tasks.named<CreateStartScripts>("startScripts").get().outputDir
    classpath = tasks.named<CreateStartScripts>("startScripts").get().classpath
}

tasks.named("startScripts") {
    dependsOn(bentoServiceClientStartScripts)
}
