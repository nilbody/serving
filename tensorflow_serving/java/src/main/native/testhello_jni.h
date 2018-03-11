//
// Created by butter on 3/11/18.
//

#ifndef SERVING_JAVA_TESTHELLO_JNI_H
#define SERVING_JAVA_TESTHELLO_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_bt_tstdemo_testHello(JNIEnv *, jclass, jstring);


#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif //SERVING_JAVA_TESTHELLO_JNI_H
