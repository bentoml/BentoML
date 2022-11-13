<?php

use Bentoml\Grpc\v1\BentoServiceClient;
use Bentoml\Grpc\v1\NDArray;
use Bentoml\Grpc\v1\Request;

require dirname(__FILE__) . '/vendor/autoload.php';

function call()
{
    $hostname = 'localhost:3000';
    $apiName = "classify";
    $to_parsed = array("3.5", "2.4", "7.8", "5.1");
    $data = array_map("floatval", $to_parsed);
    $shape = array(1, 4);
    $client = new BentoServiceClient($hostname, [
        'credentials' => Grpc\ChannelCredentials::createInsecure(),
    ]);
    $request = new Request();
    $request->setApiName($apiName);
    $payload = new NDArray();
    $payload->setShape($shape);
    $payload->setFloatValues($data);
    $payload->setDtype(\Bentoml\Grpc\v1\NDArray\DType::DTYPE_FLOAT);

    list($response, $status) = $client->Call($request)->wait();
    if ($status->code !== Grpc\STATUS_OK) {
        echo "ERROR: " . $status->code . ", " . $status->details . PHP_EOL;
        exit(1);
    }
    echo $response->getMessage() . PHP_EOL;
}

call();
