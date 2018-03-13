package(default_visibility = [
    "//tensorflow_serving/java:__pkg__",
])

load("//tensorflow_serving/java:build_defs.bzl", "tf_copts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_library")

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

SUPPORTED_TENSORFLOW_OPS = [
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]

TENSORFLOW_DEPS = [
    "@org_tensorflow//tensorflow/core:tensorflow",
    "//tensorflow_serving/servables/tensorflow:classification_service",
    "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
    "//tensorflow_serving/servables/tensorflow:multi_inference",
    "//tensorflow_serving/servables/tensorflow:regression_service",
    "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
    "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter",
    "//tensorflow_serving/servables/tensorflow:predict_impl",
]

tf_cuda_library(
    name = "native",
    srcs = glob(["*.cc"]) + [
        ":jni.h",
        ":jni_md.h",
    ],
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    includes = ["."],
    deps = [
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "@protobuf_archive//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform/cloud:gcs_file_system",
        "@org_tensorflow//tensorflow/core/platform/hadoop:hadoop_file_system",
        "@org_tensorflow//tensorflow/core/platform/s3:s3_file_system",
        "//tensorflow_serving/apis:prediction_service_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "@grpc//:grpc++_unsecure",
    ] + TENSORFLOW_DEPS + SUPPORTED_TENSORFLOW_OPS,
    alwayslink = 1,
)
