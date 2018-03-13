package(default_visibility = ["//visibility:private"])

load(":build_defs.bzl", "JAVACOPTS")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

java_library(
    name = "java_serving",
    srcs = [
        ":java_sources",
    ],
    data = [":libtensorflow_jni"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "java_sources",
    srcs = glob([
        "src/main/java/com/bt/*.java",
        "src/main/java/com/bt/**/*.java",
    ]),
    visibility = [
        "//tensorflow_serving/java:__pkg__",
    ],
)

filegroup(
    name = "libtensorflow_jni",
    srcs = [":libtensorflow_jni.so"],
    visibility = ["//visibility:public"],
)

LINKER_VERSION_SCRIPT = ":config/version_script.lds"

LINKER_EXPORTED_SYMBOLS = ":config/exported_symbols.lds"

tf_cc_binary(
    name = "libtensorflow_jni.so",
    # Set linker options to strip out anything except the JNI
    # symbols from the library. This reduces the size of the library
    # considerably (~50% as of January 2017).
    linkopts = select({
        ":debug": [],  # Disable all custom linker options in debug mode
        #        "//tensorflow:darwin": [
        #            "-Wl,-exported_symbols_list",  # This line must be directly followed by LINKER_EXPORTED_SYMBOLS
        #            LINKER_EXPORTED_SYMBOLS,
        #        ],
        #        "//tensorflow:windows": [],
        #        "//tensorflow:windows_msvc": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by LINKER_VERSION_SCRIPT
            LINKER_VERSION_SCRIPT,
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//tensorflow_serving/java/src/main/native",
        LINKER_VERSION_SCRIPT,
        LINKER_EXPORTED_SYMBOLS,
    ],
)

genrule(
    name = "pom",
    outs = ["pom.xml"],
    cmd = "$(location generate_pom) >$@",
    output_to_bindir = 1,
    tools = [":generate_pom"],  # + tf_binary_additional_srcs(),
)

tf_cc_binary(
    name = "generate_pom",
    srcs = ["generate_pom.cc"],
    deps = ["//tensorflow_serving/model_servers:tensorflow_model_server"],
)
