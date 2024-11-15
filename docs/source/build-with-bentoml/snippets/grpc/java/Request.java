import java.util.*;

int shape[] = { 1, 4 };
Iterable<Integer> shapeIterable = convert(shape);
Float array[] = { 3.5f, 2.4f, 7.8f, 5.1f };
Iterable<Float> arrayIterable = Arrays.asList(array);

NDArray.Builder builder = NDArray.newBuilder().addAllShape(shapeIterable).addAllFloatValues(arrayIterable).setDtype(NDArray.DType.DTYPE_FLOAT);

Request req = Request.newBuilder().setApiName(apiName).setNdarray(builder).build();
