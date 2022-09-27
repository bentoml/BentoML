val shape: List<Int> = listOf(1, 4)
val data: List<Float> = listOf(3.5f, 2.4f, 7.8f, 5.1f)

val ndarray = NDArray.newBuilder().addAllShape(shape).addAllFloatValues(data).build()
val req = Request.newBuilder().setApiName(apiName).setNdarray(ndarray).build()
