package tensorflow.serving;

import io.grpc.stub.ClientCalls;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 * <pre>
 * open source marker; do not remove
 * PredictionService provides access to machine-learned models loaded by
 * model_servers.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.7.1-SNAPSHOT)",
    comments = "Source: prediction_service.proto")
public final class PredictionServiceGrpc {

  private PredictionServiceGrpc() {}

  public static final String SERVICE_NAME = "tensorflow.serving.PredictionService";

  // Static method descriptors that strictly reflect the proto.
  @io.grpc.ExperimentalApi("https://github.com/grpc/grpc-java/issues/1901")
  public static final io.grpc.MethodDescriptor<Predict.PredictRequest,
      Predict.PredictResponse> METHOD_PREDICT =
      io.grpc.MethodDescriptor.<Predict.PredictRequest, Predict.PredictResponse>newBuilder()
          .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
          .setFullMethodName(generateFullMethodName(
              "tensorflow.serving.PredictionService", "Predict"))
          .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              Predict.PredictRequest.getDefaultInstance()))
          .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              Predict.PredictResponse.getDefaultInstance()))
          .setSchemaDescriptor(new PredictionServiceMethodDescriptorSupplier("Predict"))
          .build();

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static PredictionServiceStub newStub(io.grpc.Channel channel) {
    return new PredictionServiceStub(channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static PredictionServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    return new PredictionServiceBlockingStub(channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static PredictionServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    return new PredictionServiceFutureStub(channel);
  }

  /**
   * <pre>
   * open source marker; do not remove
   * PredictionService provides access to machine-learned models loaded by
   * model_servers.
   * </pre>
   */
  public static abstract class PredictionServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     * Predict -- provides access to loaded TensorFlow model.
     * </pre>
     */
    public void predict(Predict.PredictRequest request,
        io.grpc.stub.StreamObserver<Predict.PredictResponse> responseObserver) {
      asyncUnimplementedUnaryCall(METHOD_PREDICT, responseObserver);
    }

    @Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            METHOD_PREDICT,
            asyncUnaryCall(
              new MethodHandlers<
                Predict.PredictRequest,
                Predict.PredictResponse>(
                  this, METHODID_PREDICT)))
          .build();
    }
  }

  /**
   * <pre>
   * open source marker; do not remove
   * PredictionService provides access to machine-learned models loaded by
   * model_servers.
   * </pre>
   */
  public static final class PredictionServiceStub extends io.grpc.stub.AbstractStub<PredictionServiceStub> {
    private PredictionServiceStub(io.grpc.Channel channel) {
      super(channel);
    }

    private PredictionServiceStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PredictionServiceStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new PredictionServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * Predict -- provides access to loaded TensorFlow model.
     * </pre>
     */
    public void predict(Predict.PredictRequest request,
        io.grpc.stub.StreamObserver<Predict.PredictResponse> responseObserver) {
      ClientCalls.asyncUnaryCall(
          getChannel().newCall(METHOD_PREDICT, getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * open source marker; do not remove
   * PredictionService provides access to machine-learned models loaded by
   * model_servers.
   * </pre>
   */
  public static final class PredictionServiceBlockingStub extends io.grpc.stub.AbstractStub<PredictionServiceBlockingStub> {
    private PredictionServiceBlockingStub(io.grpc.Channel channel) {
      super(channel);
    }

    private PredictionServiceBlockingStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PredictionServiceBlockingStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new PredictionServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * Predict -- provides access to loaded TensorFlow model.
     * </pre>
     */
    public Predict.PredictResponse predict(Predict.PredictRequest request) {
      return blockingUnaryCall(
          getChannel(), METHOD_PREDICT, getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * open source marker; do not remove
   * PredictionService provides access to machine-learned models loaded by
   * model_servers.
   * </pre>
   */
  public static final class PredictionServiceFutureStub extends io.grpc.stub.AbstractStub<PredictionServiceFutureStub> {
    private PredictionServiceFutureStub(io.grpc.Channel channel) {
      super(channel);
    }

    private PredictionServiceFutureStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @Override
    protected PredictionServiceFutureStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new PredictionServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * Predict -- provides access to loaded TensorFlow model.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<Predict.PredictResponse> predict(
        Predict.PredictRequest request) {
      return futureUnaryCall(
          getChannel().newCall(METHOD_PREDICT, getCallOptions()), request);
    }
  }

  private static final int METHODID_PREDICT = 0;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final PredictionServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(PredictionServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_PREDICT:
          serviceImpl.predict((Predict.PredictRequest) request,
              (io.grpc.stub.StreamObserver<Predict.PredictResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @Override
    @SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class PredictionServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    PredictionServiceBaseDescriptorSupplier() {}

    @Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return PredictionServiceOuterClass.getDescriptor();
    }

    @Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("PredictionService");
    }
  }

  private static final class PredictionServiceFileDescriptorSupplier
      extends PredictionServiceBaseDescriptorSupplier {
    PredictionServiceFileDescriptorSupplier() {}
  }

  private static final class PredictionServiceMethodDescriptorSupplier
      extends PredictionServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    PredictionServiceMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (PredictionServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new PredictionServiceFileDescriptorSupplier())
              .addMethod(METHOD_PREDICT)
              .build();
        }
      }
    }
    return result;
  }
}
