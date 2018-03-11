package com.bt.tstdemo;

public class SayHello {

    public native void testHello();

    public static void main(String[] args){
        //加载C文件
        System.loadLibrary("SayHello");
        SayHello jniDemo = new SayHello();
        jniDemo.testHello();
    }

}