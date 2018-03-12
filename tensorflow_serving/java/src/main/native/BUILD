package(default_visibility = [
    "//tensorflow/java:__pkg__",
])

load("//tensorflow_serving/java:build_defs.bzl", "tf_copts")

genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)


tf_cuda_library(
    name = "native",
    srcs = glob(["*.cc"]) + [
            ":jni.h",
            ":jni_md.h",
         ],
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    includes = ["."],
    deps = ["//tensorflow_serving/model_servers:tensorflow_model_server"],
    alwayslink = 1,
)
