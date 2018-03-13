//
// Created by butter on 3/11/18.
//

#include <string.h>
#include <iostream>
#include "tensorflow_serving/java/src/main/native/testhello_jni.h"
#include <string>


JNIEXPORT jstring JNICALL Java_com_bt_tstdemo_testHello(JNIEnv* env, jclass clz, jstring str) {
    const char* cname = env->GetStringUTFChars(name, nullptr);
    std::string c = "_InnerC";
    const char* cstr = env->GetStringUTFChars(str, nullptr);
    std::string ccstr = std::string(cstr);
    std::string r = c += str;
    return env->NewStringUTF(r);
}
