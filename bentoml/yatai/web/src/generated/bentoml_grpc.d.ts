import * as $protobuf from "protobufjs";
/** Namespace bentoml. */
export namespace bentoml {

    /** Properties of a DeploymentSpec. */
    interface IDeploymentSpec {

        /** DeploymentSpec bento_name */
        bento_name?: (string|null);

        /** DeploymentSpec bento_version */
        bento_version?: (string|null);

        /** DeploymentSpec operator */
        operator?: (bentoml.DeploymentSpec.DeploymentOperator|null);

        /** DeploymentSpec custom_operator_config */
        custom_operator_config?: (bentoml.DeploymentSpec.ICustomOperatorConfig|null);

        /** DeploymentSpec sagemaker_operator_config */
        sagemaker_operator_config?: (bentoml.DeploymentSpec.ISageMakerOperatorConfig|null);

        /** DeploymentSpec aws_lambda_operator_config */
        aws_lambda_operator_config?: (bentoml.DeploymentSpec.IAwsLambdaOperatorConfig|null);

        /** DeploymentSpec azure_functions_operator_config */
        azure_functions_operator_config?: (bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig|null);
    }

    /** Represents a DeploymentSpec. */
    class DeploymentSpec implements IDeploymentSpec {

        /**
         * Constructs a new DeploymentSpec.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeploymentSpec);

        /** DeploymentSpec bento_name. */
        public bento_name: string;

        /** DeploymentSpec bento_version. */
        public bento_version: string;

        /** DeploymentSpec operator. */
        public operator: bentoml.DeploymentSpec.DeploymentOperator;

        /** DeploymentSpec custom_operator_config. */
        public custom_operator_config?: (bentoml.DeploymentSpec.ICustomOperatorConfig|null);

        /** DeploymentSpec sagemaker_operator_config. */
        public sagemaker_operator_config?: (bentoml.DeploymentSpec.ISageMakerOperatorConfig|null);

        /** DeploymentSpec aws_lambda_operator_config. */
        public aws_lambda_operator_config?: (bentoml.DeploymentSpec.IAwsLambdaOperatorConfig|null);

        /** DeploymentSpec azure_functions_operator_config. */
        public azure_functions_operator_config?: (bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig|null);

        /** DeploymentSpec deployment_operator_config. */
        public deployment_operator_config?: ("custom_operator_config"|"sagemaker_operator_config"|"aws_lambda_operator_config"|"azure_functions_operator_config");

        /**
         * Creates a new DeploymentSpec instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DeploymentSpec instance
         */
        public static create(properties?: bentoml.IDeploymentSpec): bentoml.DeploymentSpec;

        /**
         * Encodes the specified DeploymentSpec message. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param message DeploymentSpec message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeploymentSpec, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentSpec message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param message DeploymentSpec message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeploymentSpec, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec;

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec;

        /**
         * Verifies a DeploymentSpec message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DeploymentSpec
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec;

        /**
         * Creates a plain object from a DeploymentSpec message. Also converts values to other types if specified.
         * @param message DeploymentSpec
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DeploymentSpec, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentSpec to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace DeploymentSpec {

        /** DeploymentOperator enum. */
        enum DeploymentOperator {
            UNSET = 0,
            CUSTOM = 1,
            AWS_SAGEMAKER = 2,
            AWS_LAMBDA = 3,
            AZURE_FUNCTIONS = 4
        }

        /** Properties of a CustomOperatorConfig. */
        interface ICustomOperatorConfig {

            /** CustomOperatorConfig name */
            name?: (string|null);

            /** CustomOperatorConfig config */
            config?: (google.protobuf.IStruct|null);
        }

        /** Represents a CustomOperatorConfig. */
        class CustomOperatorConfig implements ICustomOperatorConfig {

            /**
             * Constructs a new CustomOperatorConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.ICustomOperatorConfig);

            /** CustomOperatorConfig name. */
            public name: string;

            /** CustomOperatorConfig config. */
            public config?: (google.protobuf.IStruct|null);

            /**
             * Creates a new CustomOperatorConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CustomOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.ICustomOperatorConfig): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Encodes the specified CustomOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param message CustomOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.DeploymentSpec.ICustomOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified CustomOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param message CustomOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.ICustomOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Verifies a CustomOperatorConfig message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns CustomOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Creates a plain object from a CustomOperatorConfig message. Also converts values to other types if specified.
             * @param message CustomOperatorConfig
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.CustomOperatorConfig, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this CustomOperatorConfig to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a SageMakerOperatorConfig. */
        interface ISageMakerOperatorConfig {

            /** SageMakerOperatorConfig region */
            region?: (string|null);

            /** SageMakerOperatorConfig instance_type */
            instance_type?: (string|null);

            /** SageMakerOperatorConfig instance_count */
            instance_count?: (number|null);

            /** SageMakerOperatorConfig api_name */
            api_name?: (string|null);

            /** SageMakerOperatorConfig num_of_gunicorn_workers_per_instance */
            num_of_gunicorn_workers_per_instance?: (number|null);

            /** SageMakerOperatorConfig timeout */
            timeout?: (number|null);
        }

        /** Represents a SageMakerOperatorConfig. */
        class SageMakerOperatorConfig implements ISageMakerOperatorConfig {

            /**
             * Constructs a new SageMakerOperatorConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.ISageMakerOperatorConfig);

            /** SageMakerOperatorConfig region. */
            public region: string;

            /** SageMakerOperatorConfig instance_type. */
            public instance_type: string;

            /** SageMakerOperatorConfig instance_count. */
            public instance_count: number;

            /** SageMakerOperatorConfig api_name. */
            public api_name: string;

            /** SageMakerOperatorConfig num_of_gunicorn_workers_per_instance. */
            public num_of_gunicorn_workers_per_instance: number;

            /** SageMakerOperatorConfig timeout. */
            public timeout: number;

            /**
             * Creates a new SageMakerOperatorConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SageMakerOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.ISageMakerOperatorConfig): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Encodes the specified SageMakerOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param message SageMakerOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.DeploymentSpec.ISageMakerOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified SageMakerOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param message SageMakerOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.ISageMakerOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Verifies a SageMakerOperatorConfig message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns SageMakerOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Creates a plain object from a SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @param message SageMakerOperatorConfig
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.SageMakerOperatorConfig, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this SageMakerOperatorConfig to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of an AwsLambdaOperatorConfig. */
        interface IAwsLambdaOperatorConfig {

            /** AwsLambdaOperatorConfig region */
            region?: (string|null);

            /** AwsLambdaOperatorConfig api_name */
            api_name?: (string|null);

            /** AwsLambdaOperatorConfig memory_size */
            memory_size?: (number|null);

            /** AwsLambdaOperatorConfig timeout */
            timeout?: (number|null);
        }

        /** Represents an AwsLambdaOperatorConfig. */
        class AwsLambdaOperatorConfig implements IAwsLambdaOperatorConfig {

            /**
             * Constructs a new AwsLambdaOperatorConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.IAwsLambdaOperatorConfig);

            /** AwsLambdaOperatorConfig region. */
            public region: string;

            /** AwsLambdaOperatorConfig api_name. */
            public api_name: string;

            /** AwsLambdaOperatorConfig memory_size. */
            public memory_size: number;

            /** AwsLambdaOperatorConfig timeout. */
            public timeout: number;

            /**
             * Creates a new AwsLambdaOperatorConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AwsLambdaOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.IAwsLambdaOperatorConfig): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Encodes the specified AwsLambdaOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param message AwsLambdaOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.DeploymentSpec.IAwsLambdaOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AwsLambdaOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param message AwsLambdaOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.IAwsLambdaOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Verifies an AwsLambdaOperatorConfig message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AwsLambdaOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Creates a plain object from an AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @param message AwsLambdaOperatorConfig
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.AwsLambdaOperatorConfig, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AwsLambdaOperatorConfig to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of an AzureFunctionsOperatorConfig. */
        interface IAzureFunctionsOperatorConfig {

            /** AzureFunctionsOperatorConfig location */
            location?: (string|null);

            /** AzureFunctionsOperatorConfig premium_plan_sku */
            premium_plan_sku?: (string|null);

            /** AzureFunctionsOperatorConfig min_instances */
            min_instances?: (number|null);

            /** AzureFunctionsOperatorConfig max_burst */
            max_burst?: (number|null);

            /** AzureFunctionsOperatorConfig function_auth_level */
            function_auth_level?: (string|null);
        }

        /** Represents an AzureFunctionsOperatorConfig. */
        class AzureFunctionsOperatorConfig implements IAzureFunctionsOperatorConfig {

            /**
             * Constructs a new AzureFunctionsOperatorConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig);

            /** AzureFunctionsOperatorConfig location. */
            public location: string;

            /** AzureFunctionsOperatorConfig premium_plan_sku. */
            public premium_plan_sku: string;

            /** AzureFunctionsOperatorConfig min_instances. */
            public min_instances: number;

            /** AzureFunctionsOperatorConfig max_burst. */
            public max_burst: number;

            /** AzureFunctionsOperatorConfig function_auth_level. */
            public function_auth_level: string;

            /**
             * Creates a new AzureFunctionsOperatorConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AzureFunctionsOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param message AzureFunctionsOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param message AzureFunctionsOperatorConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Verifies an AzureFunctionsOperatorConfig message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AzureFunctionsOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Creates a plain object from an AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @param message AzureFunctionsOperatorConfig
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AzureFunctionsOperatorConfig to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    /** Properties of a DeploymentState. */
    interface IDeploymentState {

        /** DeploymentState state */
        state?: (bentoml.DeploymentState.State|null);

        /** DeploymentState error_message */
        error_message?: (string|null);

        /** DeploymentState info_json */
        info_json?: (string|null);

        /** DeploymentState timestamp */
        timestamp?: (google.protobuf.ITimestamp|null);
    }

    /** Represents a DeploymentState. */
    class DeploymentState implements IDeploymentState {

        /**
         * Constructs a new DeploymentState.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeploymentState);

        /** DeploymentState state. */
        public state: bentoml.DeploymentState.State;

        /** DeploymentState error_message. */
        public error_message: string;

        /** DeploymentState info_json. */
        public info_json: string;

        /** DeploymentState timestamp. */
        public timestamp?: (google.protobuf.ITimestamp|null);

        /**
         * Creates a new DeploymentState instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DeploymentState instance
         */
        public static create(properties?: bentoml.IDeploymentState): bentoml.DeploymentState;

        /**
         * Encodes the specified DeploymentState message. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param message DeploymentState message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeploymentState, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentState message, length delimited. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param message DeploymentState message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeploymentState, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentState message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentState;

        /**
         * Decodes a DeploymentState message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentState;

        /**
         * Verifies a DeploymentState message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DeploymentState
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentState;

        /**
         * Creates a plain object from a DeploymentState message. Also converts values to other types if specified.
         * @param message DeploymentState
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DeploymentState, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentState to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace DeploymentState {

        /** State enum. */
        enum State {
            PENDING = 0,
            RUNNING = 1,
            SUCCEEDED = 2,
            FAILED = 3,
            UNKNOWN = 4,
            COMPLETED = 5,
            CRASH_LOOP_BACK_OFF = 6,
            ERROR = 7,
            INACTIVATED = 8
        }
    }

    /** Properties of a Deployment. */
    interface IDeployment {

        /** Deployment namespace */
        namespace?: (string|null);

        /** Deployment name */
        name?: (string|null);

        /** Deployment spec */
        spec?: (bentoml.IDeploymentSpec|null);

        /** Deployment state */
        state?: (bentoml.IDeploymentState|null);

        /** Deployment annotations */
        annotations?: ({ [k: string]: string }|null);

        /** Deployment labels */
        labels?: ({ [k: string]: string }|null);

        /** Deployment created_at */
        created_at?: (google.protobuf.ITimestamp|null);

        /** Deployment last_updated_at */
        last_updated_at?: (google.protobuf.ITimestamp|null);
    }

    /** Represents a Deployment. */
    class Deployment implements IDeployment {

        /**
         * Constructs a new Deployment.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeployment);

        /** Deployment namespace. */
        public namespace: string;

        /** Deployment name. */
        public name: string;

        /** Deployment spec. */
        public spec?: (bentoml.IDeploymentSpec|null);

        /** Deployment state. */
        public state?: (bentoml.IDeploymentState|null);

        /** Deployment annotations. */
        public annotations: { [k: string]: string };

        /** Deployment labels. */
        public labels: { [k: string]: string };

        /** Deployment created_at. */
        public created_at?: (google.protobuf.ITimestamp|null);

        /** Deployment last_updated_at. */
        public last_updated_at?: (google.protobuf.ITimestamp|null);

        /**
         * Creates a new Deployment instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Deployment instance
         */
        public static create(properties?: bentoml.IDeployment): bentoml.Deployment;

        /**
         * Encodes the specified Deployment message. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param message Deployment message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeployment, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Deployment message, length delimited. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param message Deployment message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeployment, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Deployment message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Deployment;

        /**
         * Decodes a Deployment message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Deployment;

        /**
         * Verifies a Deployment message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Deployment
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Deployment;

        /**
         * Creates a plain object from a Deployment message. Also converts values to other types if specified.
         * @param message Deployment
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.Deployment, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Deployment to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DeploymentStatus. */
    interface IDeploymentStatus {

        /** DeploymentStatus state */
        state?: (bentoml.IDeploymentState|null);
    }

    /** Represents a DeploymentStatus. */
    class DeploymentStatus implements IDeploymentStatus {

        /**
         * Constructs a new DeploymentStatus.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeploymentStatus);

        /** DeploymentStatus state. */
        public state?: (bentoml.IDeploymentState|null);

        /**
         * Creates a new DeploymentStatus instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DeploymentStatus instance
         */
        public static create(properties?: bentoml.IDeploymentStatus): bentoml.DeploymentStatus;

        /**
         * Encodes the specified DeploymentStatus message. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param message DeploymentStatus message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeploymentStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentStatus message, length delimited. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param message DeploymentStatus message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeploymentStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentStatus;

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentStatus;

        /**
         * Verifies a DeploymentStatus message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DeploymentStatus
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentStatus;

        /**
         * Creates a plain object from a DeploymentStatus message. Also converts values to other types if specified.
         * @param message DeploymentStatus
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DeploymentStatus, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentStatus to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an ApplyDeploymentRequest. */
    interface IApplyDeploymentRequest {

        /** ApplyDeploymentRequest deployment */
        deployment?: (bentoml.IDeployment|null);
    }

    /** Represents an ApplyDeploymentRequest. */
    class ApplyDeploymentRequest implements IApplyDeploymentRequest {

        /**
         * Constructs a new ApplyDeploymentRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IApplyDeploymentRequest);

        /** ApplyDeploymentRequest deployment. */
        public deployment?: (bentoml.IDeployment|null);

        /**
         * Creates a new ApplyDeploymentRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ApplyDeploymentRequest instance
         */
        public static create(properties?: bentoml.IApplyDeploymentRequest): bentoml.ApplyDeploymentRequest;

        /**
         * Encodes the specified ApplyDeploymentRequest message. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param message ApplyDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IApplyDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ApplyDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param message ApplyDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IApplyDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ApplyDeploymentRequest;

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ApplyDeploymentRequest;

        /**
         * Verifies an ApplyDeploymentRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ApplyDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ApplyDeploymentRequest;

        /**
         * Creates a plain object from an ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @param message ApplyDeploymentRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ApplyDeploymentRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ApplyDeploymentRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an ApplyDeploymentResponse. */
    interface IApplyDeploymentResponse {

        /** ApplyDeploymentResponse status */
        status?: (bentoml.IStatus|null);

        /** ApplyDeploymentResponse deployment */
        deployment?: (bentoml.IDeployment|null);
    }

    /** Represents an ApplyDeploymentResponse. */
    class ApplyDeploymentResponse implements IApplyDeploymentResponse {

        /**
         * Constructs a new ApplyDeploymentResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IApplyDeploymentResponse);

        /** ApplyDeploymentResponse status. */
        public status?: (bentoml.IStatus|null);

        /** ApplyDeploymentResponse deployment. */
        public deployment?: (bentoml.IDeployment|null);

        /**
         * Creates a new ApplyDeploymentResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ApplyDeploymentResponse instance
         */
        public static create(properties?: bentoml.IApplyDeploymentResponse): bentoml.ApplyDeploymentResponse;

        /**
         * Encodes the specified ApplyDeploymentResponse message. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param message ApplyDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IApplyDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ApplyDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param message ApplyDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IApplyDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ApplyDeploymentResponse;

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ApplyDeploymentResponse;

        /**
         * Verifies an ApplyDeploymentResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ApplyDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ApplyDeploymentResponse;

        /**
         * Creates a plain object from an ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @param message ApplyDeploymentResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ApplyDeploymentResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ApplyDeploymentResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DeleteDeploymentRequest. */
    interface IDeleteDeploymentRequest {

        /** DeleteDeploymentRequest deployment_name */
        deployment_name?: (string|null);

        /** DeleteDeploymentRequest namespace */
        namespace?: (string|null);

        /** DeleteDeploymentRequest force_delete */
        force_delete?: (boolean|null);
    }

    /** Represents a DeleteDeploymentRequest. */
    class DeleteDeploymentRequest implements IDeleteDeploymentRequest {

        /**
         * Constructs a new DeleteDeploymentRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeleteDeploymentRequest);

        /** DeleteDeploymentRequest deployment_name. */
        public deployment_name: string;

        /** DeleteDeploymentRequest namespace. */
        public namespace: string;

        /** DeleteDeploymentRequest force_delete. */
        public force_delete: boolean;

        /**
         * Creates a new DeleteDeploymentRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DeleteDeploymentRequest instance
         */
        public static create(properties?: bentoml.IDeleteDeploymentRequest): bentoml.DeleteDeploymentRequest;

        /**
         * Encodes the specified DeleteDeploymentRequest message. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param message DeleteDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeleteDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeleteDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param message DeleteDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeleteDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeleteDeploymentRequest;

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeleteDeploymentRequest;

        /**
         * Verifies a DeleteDeploymentRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DeleteDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeleteDeploymentRequest;

        /**
         * Creates a plain object from a DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @param message DeleteDeploymentRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DeleteDeploymentRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DeleteDeploymentRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DeleteDeploymentResponse. */
    interface IDeleteDeploymentResponse {

        /** DeleteDeploymentResponse status */
        status?: (bentoml.IStatus|null);
    }

    /** Represents a DeleteDeploymentResponse. */
    class DeleteDeploymentResponse implements IDeleteDeploymentResponse {

        /**
         * Constructs a new DeleteDeploymentResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDeleteDeploymentResponse);

        /** DeleteDeploymentResponse status. */
        public status?: (bentoml.IStatus|null);

        /**
         * Creates a new DeleteDeploymentResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DeleteDeploymentResponse instance
         */
        public static create(properties?: bentoml.IDeleteDeploymentResponse): bentoml.DeleteDeploymentResponse;

        /**
         * Encodes the specified DeleteDeploymentResponse message. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param message DeleteDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDeleteDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeleteDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param message DeleteDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDeleteDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeleteDeploymentResponse;

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeleteDeploymentResponse;

        /**
         * Verifies a DeleteDeploymentResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DeleteDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeleteDeploymentResponse;

        /**
         * Creates a plain object from a DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @param message DeleteDeploymentResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DeleteDeploymentResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DeleteDeploymentResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GetDeploymentRequest. */
    interface IGetDeploymentRequest {

        /** GetDeploymentRequest deployment_name */
        deployment_name?: (string|null);

        /** GetDeploymentRequest namespace */
        namespace?: (string|null);
    }

    /** Represents a GetDeploymentRequest. */
    class GetDeploymentRequest implements IGetDeploymentRequest {

        /**
         * Constructs a new GetDeploymentRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IGetDeploymentRequest);

        /** GetDeploymentRequest deployment_name. */
        public deployment_name: string;

        /** GetDeploymentRequest namespace. */
        public namespace: string;

        /**
         * Creates a new GetDeploymentRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GetDeploymentRequest instance
         */
        public static create(properties?: bentoml.IGetDeploymentRequest): bentoml.GetDeploymentRequest;

        /**
         * Encodes the specified GetDeploymentRequest message. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param message GetDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IGetDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param message GetDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IGetDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetDeploymentRequest;

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetDeploymentRequest;

        /**
         * Verifies a GetDeploymentRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GetDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetDeploymentRequest;

        /**
         * Creates a plain object from a GetDeploymentRequest message. Also converts values to other types if specified.
         * @param message GetDeploymentRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.GetDeploymentRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GetDeploymentRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GetDeploymentResponse. */
    interface IGetDeploymentResponse {

        /** GetDeploymentResponse status */
        status?: (bentoml.IStatus|null);

        /** GetDeploymentResponse deployment */
        deployment?: (bentoml.IDeployment|null);
    }

    /** Represents a GetDeploymentResponse. */
    class GetDeploymentResponse implements IGetDeploymentResponse {

        /**
         * Constructs a new GetDeploymentResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IGetDeploymentResponse);

        /** GetDeploymentResponse status. */
        public status?: (bentoml.IStatus|null);

        /** GetDeploymentResponse deployment. */
        public deployment?: (bentoml.IDeployment|null);

        /**
         * Creates a new GetDeploymentResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GetDeploymentResponse instance
         */
        public static create(properties?: bentoml.IGetDeploymentResponse): bentoml.GetDeploymentResponse;

        /**
         * Encodes the specified GetDeploymentResponse message. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param message GetDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IGetDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param message GetDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IGetDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetDeploymentResponse;

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetDeploymentResponse;

        /**
         * Verifies a GetDeploymentResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GetDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetDeploymentResponse;

        /**
         * Creates a plain object from a GetDeploymentResponse message. Also converts values to other types if specified.
         * @param message GetDeploymentResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.GetDeploymentResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GetDeploymentResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DescribeDeploymentRequest. */
    interface IDescribeDeploymentRequest {

        /** DescribeDeploymentRequest deployment_name */
        deployment_name?: (string|null);

        /** DescribeDeploymentRequest namespace */
        namespace?: (string|null);
    }

    /** Represents a DescribeDeploymentRequest. */
    class DescribeDeploymentRequest implements IDescribeDeploymentRequest {

        /**
         * Constructs a new DescribeDeploymentRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDescribeDeploymentRequest);

        /** DescribeDeploymentRequest deployment_name. */
        public deployment_name: string;

        /** DescribeDeploymentRequest namespace. */
        public namespace: string;

        /**
         * Creates a new DescribeDeploymentRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DescribeDeploymentRequest instance
         */
        public static create(properties?: bentoml.IDescribeDeploymentRequest): bentoml.DescribeDeploymentRequest;

        /**
         * Encodes the specified DescribeDeploymentRequest message. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param message DescribeDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDescribeDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DescribeDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param message DescribeDeploymentRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDescribeDeploymentRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DescribeDeploymentRequest;

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DescribeDeploymentRequest;

        /**
         * Verifies a DescribeDeploymentRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DescribeDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DescribeDeploymentRequest;

        /**
         * Creates a plain object from a DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @param message DescribeDeploymentRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DescribeDeploymentRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DescribeDeploymentRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DescribeDeploymentResponse. */
    interface IDescribeDeploymentResponse {

        /** DescribeDeploymentResponse status */
        status?: (bentoml.IStatus|null);

        /** DescribeDeploymentResponse state */
        state?: (bentoml.IDeploymentState|null);
    }

    /** Represents a DescribeDeploymentResponse. */
    class DescribeDeploymentResponse implements IDescribeDeploymentResponse {

        /**
         * Constructs a new DescribeDeploymentResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDescribeDeploymentResponse);

        /** DescribeDeploymentResponse status. */
        public status?: (bentoml.IStatus|null);

        /** DescribeDeploymentResponse state. */
        public state?: (bentoml.IDeploymentState|null);

        /**
         * Creates a new DescribeDeploymentResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DescribeDeploymentResponse instance
         */
        public static create(properties?: bentoml.IDescribeDeploymentResponse): bentoml.DescribeDeploymentResponse;

        /**
         * Encodes the specified DescribeDeploymentResponse message. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param message DescribeDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDescribeDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DescribeDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param message DescribeDeploymentResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDescribeDeploymentResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DescribeDeploymentResponse;

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DescribeDeploymentResponse;

        /**
         * Verifies a DescribeDeploymentResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DescribeDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DescribeDeploymentResponse;

        /**
         * Creates a plain object from a DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @param message DescribeDeploymentResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DescribeDeploymentResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DescribeDeploymentResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a ListDeploymentsRequest. */
    interface IListDeploymentsRequest {

        /** ListDeploymentsRequest namespace */
        namespace?: (string|null);

        /** ListDeploymentsRequest offset */
        offset?: (number|null);

        /** ListDeploymentsRequest limit */
        limit?: (number|null);

        /** ListDeploymentsRequest operator */
        operator?: (bentoml.DeploymentSpec.DeploymentOperator|null);

        /** ListDeploymentsRequest order_by */
        order_by?: (bentoml.ListDeploymentsRequest.SORTABLE_COLUMN|null);

        /** ListDeploymentsRequest ascending_order */
        ascending_order?: (boolean|null);

        /** ListDeploymentsRequest labels_query */
        labels_query?: (string|null);
    }

    /** Represents a ListDeploymentsRequest. */
    class ListDeploymentsRequest implements IListDeploymentsRequest {

        /**
         * Constructs a new ListDeploymentsRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IListDeploymentsRequest);

        /** ListDeploymentsRequest namespace. */
        public namespace: string;

        /** ListDeploymentsRequest offset. */
        public offset: number;

        /** ListDeploymentsRequest limit. */
        public limit: number;

        /** ListDeploymentsRequest operator. */
        public operator: bentoml.DeploymentSpec.DeploymentOperator;

        /** ListDeploymentsRequest order_by. */
        public order_by: bentoml.ListDeploymentsRequest.SORTABLE_COLUMN;

        /** ListDeploymentsRequest ascending_order. */
        public ascending_order: boolean;

        /** ListDeploymentsRequest labels_query. */
        public labels_query: string;

        /**
         * Creates a new ListDeploymentsRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ListDeploymentsRequest instance
         */
        public static create(properties?: bentoml.IListDeploymentsRequest): bentoml.ListDeploymentsRequest;

        /**
         * Encodes the specified ListDeploymentsRequest message. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param message ListDeploymentsRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IListDeploymentsRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListDeploymentsRequest message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param message ListDeploymentsRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IListDeploymentsRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListDeploymentsRequest;

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListDeploymentsRequest;

        /**
         * Verifies a ListDeploymentsRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ListDeploymentsRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListDeploymentsRequest;

        /**
         * Creates a plain object from a ListDeploymentsRequest message. Also converts values to other types if specified.
         * @param message ListDeploymentsRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ListDeploymentsRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ListDeploymentsRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace ListDeploymentsRequest {

        /** SORTABLE_COLUMN enum. */
        enum SORTABLE_COLUMN {
            created_at = 0,
            name = 1
        }
    }

    /** Properties of a ListDeploymentsResponse. */
    interface IListDeploymentsResponse {

        /** ListDeploymentsResponse status */
        status?: (bentoml.IStatus|null);

        /** ListDeploymentsResponse deployments */
        deployments?: (bentoml.IDeployment[]|null);
    }

    /** Represents a ListDeploymentsResponse. */
    class ListDeploymentsResponse implements IListDeploymentsResponse {

        /**
         * Constructs a new ListDeploymentsResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IListDeploymentsResponse);

        /** ListDeploymentsResponse status. */
        public status?: (bentoml.IStatus|null);

        /** ListDeploymentsResponse deployments. */
        public deployments: bentoml.IDeployment[];

        /**
         * Creates a new ListDeploymentsResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ListDeploymentsResponse instance
         */
        public static create(properties?: bentoml.IListDeploymentsResponse): bentoml.ListDeploymentsResponse;

        /**
         * Encodes the specified ListDeploymentsResponse message. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param message ListDeploymentsResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IListDeploymentsResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListDeploymentsResponse message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param message ListDeploymentsResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IListDeploymentsResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListDeploymentsResponse;

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListDeploymentsResponse;

        /**
         * Verifies a ListDeploymentsResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a ListDeploymentsResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ListDeploymentsResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListDeploymentsResponse;

        /**
         * Creates a plain object from a ListDeploymentsResponse message. Also converts values to other types if specified.
         * @param message ListDeploymentsResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ListDeploymentsResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ListDeploymentsResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a Status. */
    interface IStatus {

        /** Status status_code */
        status_code?: (bentoml.Status.Code|null);

        /** Status error_message */
        error_message?: (string|null);
    }

    /** Represents a Status. */
    class Status implements IStatus {

        /**
         * Constructs a new Status.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IStatus);

        /** Status status_code. */
        public status_code: bentoml.Status.Code;

        /** Status error_message. */
        public error_message: string;

        /**
         * Creates a new Status instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Status instance
         */
        public static create(properties?: bentoml.IStatus): bentoml.Status;

        /**
         * Encodes the specified Status message. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param message Status message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Status message, length delimited. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param message Status message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Status message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Status;

        /**
         * Decodes a Status message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Status;

        /**
         * Verifies a Status message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Status
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Status;

        /**
         * Creates a plain object from a Status message. Also converts values to other types if specified.
         * @param message Status
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.Status, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Status to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace Status {

        /** Code enum. */
        enum Code {
            OK = 0,
            CANCELLED = 1,
            UNKNOWN = 2,
            INVALID_ARGUMENT = 3,
            DEADLINE_EXCEEDED = 4,
            NOT_FOUND = 5,
            ALREADY_EXISTS = 6,
            PERMISSION_DENIED = 7,
            UNAUTHENTICATED = 16,
            RESOURCE_EXHAUSTED = 8,
            FAILED_PRECONDITION = 9,
            ABORTED = 10,
            OUT_OF_RANGE = 11,
            UNIMPLEMENTED = 12,
            INTERNAL = 13,
            UNAVAILABLE = 14,
            DATA_LOSS = 15,
            DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 20
        }
    }

    /** Properties of a BentoUri. */
    interface IBentoUri {

        /** BentoUri type */
        type?: (bentoml.BentoUri.StorageType|null);

        /** BentoUri uri */
        uri?: (string|null);

        /** BentoUri s3_presigned_url */
        s3_presigned_url?: (string|null);
    }

    /** Represents a BentoUri. */
    class BentoUri implements IBentoUri {

        /**
         * Constructs a new BentoUri.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IBentoUri);

        /** BentoUri type. */
        public type: bentoml.BentoUri.StorageType;

        /** BentoUri uri. */
        public uri: string;

        /** BentoUri s3_presigned_url. */
        public s3_presigned_url: string;

        /**
         * Creates a new BentoUri instance using the specified properties.
         * @param [properties] Properties to set
         * @returns BentoUri instance
         */
        public static create(properties?: bentoml.IBentoUri): bentoml.BentoUri;

        /**
         * Encodes the specified BentoUri message. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param message BentoUri message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IBentoUri, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified BentoUri message, length delimited. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param message BentoUri message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IBentoUri, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a BentoUri message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoUri;

        /**
         * Decodes a BentoUri message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoUri;

        /**
         * Verifies a BentoUri message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns BentoUri
         */
        public static fromObject(object: { [k: string]: any }): bentoml.BentoUri;

        /**
         * Creates a plain object from a BentoUri message. Also converts values to other types if specified.
         * @param message BentoUri
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.BentoUri, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this BentoUri to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace BentoUri {

        /** StorageType enum. */
        enum StorageType {
            UNSET = 0,
            LOCAL = 1,
            S3 = 2,
            GCS = 3,
            AZURE_BLOB_STORAGE = 4,
            HDFS = 5
        }
    }

    /** Properties of a BentoServiceMetadata. */
    interface IBentoServiceMetadata {

        /** BentoServiceMetadata name */
        name?: (string|null);

        /** BentoServiceMetadata version */
        version?: (string|null);

        /** BentoServiceMetadata created_at */
        created_at?: (google.protobuf.ITimestamp|null);

        /** BentoServiceMetadata env */
        env?: (bentoml.BentoServiceMetadata.IBentoServiceEnv|null);

        /** BentoServiceMetadata artifacts */
        artifacts?: (bentoml.BentoServiceMetadata.IBentoArtifact[]|null);

        /** BentoServiceMetadata apis */
        apis?: (bentoml.BentoServiceMetadata.IBentoServiceApi[]|null);
    }

    /** Represents a BentoServiceMetadata. */
    class BentoServiceMetadata implements IBentoServiceMetadata {

        /**
         * Constructs a new BentoServiceMetadata.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IBentoServiceMetadata);

        /** BentoServiceMetadata name. */
        public name: string;

        /** BentoServiceMetadata version. */
        public version: string;

        /** BentoServiceMetadata created_at. */
        public created_at?: (google.protobuf.ITimestamp|null);

        /** BentoServiceMetadata env. */
        public env?: (bentoml.BentoServiceMetadata.IBentoServiceEnv|null);

        /** BentoServiceMetadata artifacts. */
        public artifacts: bentoml.BentoServiceMetadata.IBentoArtifact[];

        /** BentoServiceMetadata apis. */
        public apis: bentoml.BentoServiceMetadata.IBentoServiceApi[];

        /**
         * Creates a new BentoServiceMetadata instance using the specified properties.
         * @param [properties] Properties to set
         * @returns BentoServiceMetadata instance
         */
        public static create(properties?: bentoml.IBentoServiceMetadata): bentoml.BentoServiceMetadata;

        /**
         * Encodes the specified BentoServiceMetadata message. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param message BentoServiceMetadata message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IBentoServiceMetadata, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified BentoServiceMetadata message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param message BentoServiceMetadata message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IBentoServiceMetadata, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata;

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata;

        /**
         * Verifies a BentoServiceMetadata message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a BentoServiceMetadata message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns BentoServiceMetadata
         */
        public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata;

        /**
         * Creates a plain object from a BentoServiceMetadata message. Also converts values to other types if specified.
         * @param message BentoServiceMetadata
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.BentoServiceMetadata, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this BentoServiceMetadata to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace BentoServiceMetadata {

        /** Properties of a BentoServiceEnv. */
        interface IBentoServiceEnv {

            /** BentoServiceEnv setup_sh */
            setup_sh?: (string|null);

            /** BentoServiceEnv conda_env */
            conda_env?: (string|null);

            /** BentoServiceEnv pip_dependencies */
            pip_dependencies?: (string|null);

            /** BentoServiceEnv python_version */
            python_version?: (string|null);

            /** BentoServiceEnv docker_base_image */
            docker_base_image?: (string|null);
        }

        /** Represents a BentoServiceEnv. */
        class BentoServiceEnv implements IBentoServiceEnv {

            /**
             * Constructs a new BentoServiceEnv.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.IBentoServiceEnv);

            /** BentoServiceEnv setup_sh. */
            public setup_sh: string;

            /** BentoServiceEnv conda_env. */
            public conda_env: string;

            /** BentoServiceEnv pip_dependencies. */
            public pip_dependencies: string;

            /** BentoServiceEnv python_version. */
            public python_version: string;

            /** BentoServiceEnv docker_base_image. */
            public docker_base_image: string;

            /**
             * Creates a new BentoServiceEnv instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BentoServiceEnv instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.IBentoServiceEnv): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Encodes the specified BentoServiceEnv message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param message BentoServiceEnv message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.IBentoServiceEnv, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoServiceEnv message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param message BentoServiceEnv message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.IBentoServiceEnv, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Verifies a BentoServiceEnv message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns BentoServiceEnv
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Creates a plain object from a BentoServiceEnv message. Also converts values to other types if specified.
             * @param message BentoServiceEnv
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoServiceEnv, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoServiceEnv to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a BentoArtifact. */
        interface IBentoArtifact {

            /** BentoArtifact name */
            name?: (string|null);

            /** BentoArtifact artifact_type */
            artifact_type?: (string|null);
        }

        /** Represents a BentoArtifact. */
        class BentoArtifact implements IBentoArtifact {

            /**
             * Constructs a new BentoArtifact.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.IBentoArtifact);

            /** BentoArtifact name. */
            public name: string;

            /** BentoArtifact artifact_type. */
            public artifact_type: string;

            /**
             * Creates a new BentoArtifact instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BentoArtifact instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.IBentoArtifact): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Encodes the specified BentoArtifact message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param message BentoArtifact message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.IBentoArtifact, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoArtifact message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param message BentoArtifact message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.IBentoArtifact, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Verifies a BentoArtifact message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns BentoArtifact
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Creates a plain object from a BentoArtifact message. Also converts values to other types if specified.
             * @param message BentoArtifact
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoArtifact, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoArtifact to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a BentoServiceApi. */
        interface IBentoServiceApi {

            /** BentoServiceApi name */
            name?: (string|null);

            /** BentoServiceApi input_type */
            input_type?: (string|null);

            /** BentoServiceApi docs */
            docs?: (string|null);

            /** BentoServiceApi input_config */
            input_config?: (google.protobuf.IStruct|null);

            /** BentoServiceApi output_config */
            output_config?: (google.protobuf.IStruct|null);

            /** BentoServiceApi output_type */
            output_type?: (string|null);

            /** BentoServiceApi mb_max_latency */
            mb_max_latency?: (number|null);

            /** BentoServiceApi mb_max_batch_size */
            mb_max_batch_size?: (number|null);
        }

        /** Represents a BentoServiceApi. */
        class BentoServiceApi implements IBentoServiceApi {

            /**
             * Constructs a new BentoServiceApi.
             * @param [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.IBentoServiceApi);

            /** BentoServiceApi name. */
            public name: string;

            /** BentoServiceApi input_type. */
            public input_type: string;

            /** BentoServiceApi docs. */
            public docs: string;

            /** BentoServiceApi input_config. */
            public input_config?: (google.protobuf.IStruct|null);

            /** BentoServiceApi output_config. */
            public output_config?: (google.protobuf.IStruct|null);

            /** BentoServiceApi output_type. */
            public output_type: string;

            /** BentoServiceApi mb_max_latency. */
            public mb_max_latency: number;

            /** BentoServiceApi mb_max_batch_size. */
            public mb_max_batch_size: number;

            /**
             * Creates a new BentoServiceApi instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BentoServiceApi instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.IBentoServiceApi): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Encodes the specified BentoServiceApi message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param message BentoServiceApi message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.IBentoServiceApi, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoServiceApi message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param message BentoServiceApi message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.IBentoServiceApi, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Verifies a BentoServiceApi message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns BentoServiceApi
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Creates a plain object from a BentoServiceApi message. Also converts values to other types if specified.
             * @param message BentoServiceApi
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoServiceApi, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoServiceApi to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    /** Properties of a Bento. */
    interface IBento {

        /** Bento name */
        name?: (string|null);

        /** Bento version */
        version?: (string|null);

        /** Bento uri */
        uri?: (bentoml.IBentoUri|null);

        /** Bento bento_service_metadata */
        bento_service_metadata?: (bentoml.IBentoServiceMetadata|null);

        /** Bento status */
        status?: (bentoml.IUploadStatus|null);
    }

    /** Represents a Bento. */
    class Bento implements IBento {

        /**
         * Constructs a new Bento.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IBento);

        /** Bento name. */
        public name: string;

        /** Bento version. */
        public version: string;

        /** Bento uri. */
        public uri?: (bentoml.IBentoUri|null);

        /** Bento bento_service_metadata. */
        public bento_service_metadata?: (bentoml.IBentoServiceMetadata|null);

        /** Bento status. */
        public status?: (bentoml.IUploadStatus|null);

        /**
         * Creates a new Bento instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Bento instance
         */
        public static create(properties?: bentoml.IBento): bentoml.Bento;

        /**
         * Encodes the specified Bento message. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param message Bento message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IBento, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Bento message, length delimited. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param message Bento message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IBento, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Bento message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Bento;

        /**
         * Decodes a Bento message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Bento;

        /**
         * Verifies a Bento message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Bento
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Bento;

        /**
         * Creates a plain object from a Bento message. Also converts values to other types if specified.
         * @param message Bento
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.Bento, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Bento to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an AddBentoRequest. */
    interface IAddBentoRequest {

        /** AddBentoRequest bento_name */
        bento_name?: (string|null);

        /** AddBentoRequest bento_version */
        bento_version?: (string|null);
    }

    /** Represents an AddBentoRequest. */
    class AddBentoRequest implements IAddBentoRequest {

        /**
         * Constructs a new AddBentoRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IAddBentoRequest);

        /** AddBentoRequest bento_name. */
        public bento_name: string;

        /** AddBentoRequest bento_version. */
        public bento_version: string;

        /**
         * Creates a new AddBentoRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns AddBentoRequest instance
         */
        public static create(properties?: bentoml.IAddBentoRequest): bentoml.AddBentoRequest;

        /**
         * Encodes the specified AddBentoRequest message. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param message AddBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IAddBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AddBentoRequest message, length delimited. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param message AddBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IAddBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.AddBentoRequest;

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.AddBentoRequest;

        /**
         * Verifies an AddBentoRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns AddBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.AddBentoRequest;

        /**
         * Creates a plain object from an AddBentoRequest message. Also converts values to other types if specified.
         * @param message AddBentoRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.AddBentoRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this AddBentoRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an AddBentoResponse. */
    interface IAddBentoResponse {

        /** AddBentoResponse status */
        status?: (bentoml.IStatus|null);

        /** AddBentoResponse uri */
        uri?: (bentoml.IBentoUri|null);
    }

    /** Represents an AddBentoResponse. */
    class AddBentoResponse implements IAddBentoResponse {

        /**
         * Constructs a new AddBentoResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IAddBentoResponse);

        /** AddBentoResponse status. */
        public status?: (bentoml.IStatus|null);

        /** AddBentoResponse uri. */
        public uri?: (bentoml.IBentoUri|null);

        /**
         * Creates a new AddBentoResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns AddBentoResponse instance
         */
        public static create(properties?: bentoml.IAddBentoResponse): bentoml.AddBentoResponse;

        /**
         * Encodes the specified AddBentoResponse message. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param message AddBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IAddBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AddBentoResponse message, length delimited. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param message AddBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IAddBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.AddBentoResponse;

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.AddBentoResponse;

        /**
         * Verifies an AddBentoResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns AddBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.AddBentoResponse;

        /**
         * Creates a plain object from an AddBentoResponse message. Also converts values to other types if specified.
         * @param message AddBentoResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.AddBentoResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this AddBentoResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an UploadStatus. */
    interface IUploadStatus {

        /** UploadStatus status */
        status?: (bentoml.UploadStatus.Status|null);

        /** UploadStatus updated_at */
        updated_at?: (google.protobuf.ITimestamp|null);

        /** UploadStatus percentage */
        percentage?: (number|null);

        /** UploadStatus error_message */
        error_message?: (string|null);
    }

    /** Represents an UploadStatus. */
    class UploadStatus implements IUploadStatus {

        /**
         * Constructs a new UploadStatus.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IUploadStatus);

        /** UploadStatus status. */
        public status: bentoml.UploadStatus.Status;

        /** UploadStatus updated_at. */
        public updated_at?: (google.protobuf.ITimestamp|null);

        /** UploadStatus percentage. */
        public percentage: number;

        /** UploadStatus error_message. */
        public error_message: string;

        /**
         * Creates a new UploadStatus instance using the specified properties.
         * @param [properties] Properties to set
         * @returns UploadStatus instance
         */
        public static create(properties?: bentoml.IUploadStatus): bentoml.UploadStatus;

        /**
         * Encodes the specified UploadStatus message. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param message UploadStatus message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IUploadStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UploadStatus message, length delimited. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param message UploadStatus message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IUploadStatus, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UploadStatus message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UploadStatus;

        /**
         * Decodes an UploadStatus message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UploadStatus;

        /**
         * Verifies an UploadStatus message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns UploadStatus
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UploadStatus;

        /**
         * Creates a plain object from an UploadStatus message. Also converts values to other types if specified.
         * @param message UploadStatus
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.UploadStatus, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this UploadStatus to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace UploadStatus {

        /** Status enum. */
        enum Status {
            UNINITIALIZED = 0,
            UPLOADING = 1,
            DONE = 2,
            ERROR = 3,
            TIMEOUT = 4
        }
    }

    /** Properties of an UpdateBentoRequest. */
    interface IUpdateBentoRequest {

        /** UpdateBentoRequest bento_name */
        bento_name?: (string|null);

        /** UpdateBentoRequest bento_version */
        bento_version?: (string|null);

        /** UpdateBentoRequest upload_status */
        upload_status?: (bentoml.IUploadStatus|null);

        /** UpdateBentoRequest service_metadata */
        service_metadata?: (bentoml.IBentoServiceMetadata|null);
    }

    /** Represents an UpdateBentoRequest. */
    class UpdateBentoRequest implements IUpdateBentoRequest {

        /**
         * Constructs a new UpdateBentoRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IUpdateBentoRequest);

        /** UpdateBentoRequest bento_name. */
        public bento_name: string;

        /** UpdateBentoRequest bento_version. */
        public bento_version: string;

        /** UpdateBentoRequest upload_status. */
        public upload_status?: (bentoml.IUploadStatus|null);

        /** UpdateBentoRequest service_metadata. */
        public service_metadata?: (bentoml.IBentoServiceMetadata|null);

        /**
         * Creates a new UpdateBentoRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns UpdateBentoRequest instance
         */
        public static create(properties?: bentoml.IUpdateBentoRequest): bentoml.UpdateBentoRequest;

        /**
         * Encodes the specified UpdateBentoRequest message. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param message UpdateBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IUpdateBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UpdateBentoRequest message, length delimited. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param message UpdateBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IUpdateBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UpdateBentoRequest;

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UpdateBentoRequest;

        /**
         * Verifies an UpdateBentoRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns UpdateBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UpdateBentoRequest;

        /**
         * Creates a plain object from an UpdateBentoRequest message. Also converts values to other types if specified.
         * @param message UpdateBentoRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.UpdateBentoRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this UpdateBentoRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of an UpdateBentoResponse. */
    interface IUpdateBentoResponse {

        /** UpdateBentoResponse status */
        status?: (bentoml.IStatus|null);
    }

    /** Represents an UpdateBentoResponse. */
    class UpdateBentoResponse implements IUpdateBentoResponse {

        /**
         * Constructs a new UpdateBentoResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IUpdateBentoResponse);

        /** UpdateBentoResponse status. */
        public status?: (bentoml.IStatus|null);

        /**
         * Creates a new UpdateBentoResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns UpdateBentoResponse instance
         */
        public static create(properties?: bentoml.IUpdateBentoResponse): bentoml.UpdateBentoResponse;

        /**
         * Encodes the specified UpdateBentoResponse message. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param message UpdateBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IUpdateBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UpdateBentoResponse message, length delimited. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param message UpdateBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IUpdateBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UpdateBentoResponse;

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UpdateBentoResponse;

        /**
         * Verifies an UpdateBentoResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns UpdateBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UpdateBentoResponse;

        /**
         * Creates a plain object from an UpdateBentoResponse message. Also converts values to other types if specified.
         * @param message UpdateBentoResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.UpdateBentoResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this UpdateBentoResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DangerouslyDeleteBentoRequest. */
    interface IDangerouslyDeleteBentoRequest {

        /** DangerouslyDeleteBentoRequest bento_name */
        bento_name?: (string|null);

        /** DangerouslyDeleteBentoRequest bento_version */
        bento_version?: (string|null);
    }

    /** Represents a DangerouslyDeleteBentoRequest. */
    class DangerouslyDeleteBentoRequest implements IDangerouslyDeleteBentoRequest {

        /**
         * Constructs a new DangerouslyDeleteBentoRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDangerouslyDeleteBentoRequest);

        /** DangerouslyDeleteBentoRequest bento_name. */
        public bento_name: string;

        /** DangerouslyDeleteBentoRequest bento_version. */
        public bento_version: string;

        /**
         * Creates a new DangerouslyDeleteBentoRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DangerouslyDeleteBentoRequest instance
         */
        public static create(properties?: bentoml.IDangerouslyDeleteBentoRequest): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDangerouslyDeleteBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDangerouslyDeleteBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Verifies a DangerouslyDeleteBentoRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DangerouslyDeleteBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @param message DangerouslyDeleteBentoRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DangerouslyDeleteBentoRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DangerouslyDeleteBentoRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a DangerouslyDeleteBentoResponse. */
    interface IDangerouslyDeleteBentoResponse {

        /** DangerouslyDeleteBentoResponse status */
        status?: (bentoml.IStatus|null);
    }

    /** Represents a DangerouslyDeleteBentoResponse. */
    class DangerouslyDeleteBentoResponse implements IDangerouslyDeleteBentoResponse {

        /**
         * Constructs a new DangerouslyDeleteBentoResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IDangerouslyDeleteBentoResponse);

        /** DangerouslyDeleteBentoResponse status. */
        public status?: (bentoml.IStatus|null);

        /**
         * Creates a new DangerouslyDeleteBentoResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns DangerouslyDeleteBentoResponse instance
         */
        public static create(properties?: bentoml.IDangerouslyDeleteBentoResponse): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IDangerouslyDeleteBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IDangerouslyDeleteBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Verifies a DangerouslyDeleteBentoResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns DangerouslyDeleteBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @param message DangerouslyDeleteBentoResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.DangerouslyDeleteBentoResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this DangerouslyDeleteBentoResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GetBentoRequest. */
    interface IGetBentoRequest {

        /** GetBentoRequest bento_name */
        bento_name?: (string|null);

        /** GetBentoRequest bento_version */
        bento_version?: (string|null);
    }

    /** Represents a GetBentoRequest. */
    class GetBentoRequest implements IGetBentoRequest {

        /**
         * Constructs a new GetBentoRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IGetBentoRequest);

        /** GetBentoRequest bento_name. */
        public bento_name: string;

        /** GetBentoRequest bento_version. */
        public bento_version: string;

        /**
         * Creates a new GetBentoRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GetBentoRequest instance
         */
        public static create(properties?: bentoml.IGetBentoRequest): bentoml.GetBentoRequest;

        /**
         * Encodes the specified GetBentoRequest message. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param message GetBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IGetBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetBentoRequest message, length delimited. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param message GetBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IGetBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetBentoRequest;

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetBentoRequest;

        /**
         * Verifies a GetBentoRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GetBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetBentoRequest;

        /**
         * Creates a plain object from a GetBentoRequest message. Also converts values to other types if specified.
         * @param message GetBentoRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.GetBentoRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GetBentoRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GetBentoResponse. */
    interface IGetBentoResponse {

        /** GetBentoResponse status */
        status?: (bentoml.IStatus|null);

        /** GetBentoResponse bento */
        bento?: (bentoml.IBento|null);
    }

    /** Represents a GetBentoResponse. */
    class GetBentoResponse implements IGetBentoResponse {

        /**
         * Constructs a new GetBentoResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IGetBentoResponse);

        /** GetBentoResponse status. */
        public status?: (bentoml.IStatus|null);

        /** GetBentoResponse bento. */
        public bento?: (bentoml.IBento|null);

        /**
         * Creates a new GetBentoResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GetBentoResponse instance
         */
        public static create(properties?: bentoml.IGetBentoResponse): bentoml.GetBentoResponse;

        /**
         * Encodes the specified GetBentoResponse message. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param message GetBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IGetBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetBentoResponse message, length delimited. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param message GetBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IGetBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetBentoResponse;

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetBentoResponse;

        /**
         * Verifies a GetBentoResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GetBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetBentoResponse;

        /**
         * Creates a plain object from a GetBentoResponse message. Also converts values to other types if specified.
         * @param message GetBentoResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.GetBentoResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GetBentoResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a ListBentoRequest. */
    interface IListBentoRequest {

        /** ListBentoRequest bento_name */
        bento_name?: (string|null);

        /** ListBentoRequest offset */
        offset?: (number|null);

        /** ListBentoRequest limit */
        limit?: (number|null);

        /** ListBentoRequest order_by */
        order_by?: (bentoml.ListBentoRequest.SORTABLE_COLUMN|null);

        /** ListBentoRequest ascending_order */
        ascending_order?: (boolean|null);
    }

    /** Represents a ListBentoRequest. */
    class ListBentoRequest implements IListBentoRequest {

        /**
         * Constructs a new ListBentoRequest.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IListBentoRequest);

        /** ListBentoRequest bento_name. */
        public bento_name: string;

        /** ListBentoRequest offset. */
        public offset: number;

        /** ListBentoRequest limit. */
        public limit: number;

        /** ListBentoRequest order_by. */
        public order_by: bentoml.ListBentoRequest.SORTABLE_COLUMN;

        /** ListBentoRequest ascending_order. */
        public ascending_order: boolean;

        /**
         * Creates a new ListBentoRequest instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ListBentoRequest instance
         */
        public static create(properties?: bentoml.IListBentoRequest): bentoml.ListBentoRequest;

        /**
         * Encodes the specified ListBentoRequest message. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param message ListBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IListBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListBentoRequest message, length delimited. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param message ListBentoRequest message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IListBentoRequest, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListBentoRequest;

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListBentoRequest;

        /**
         * Verifies a ListBentoRequest message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ListBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListBentoRequest;

        /**
         * Creates a plain object from a ListBentoRequest message. Also converts values to other types if specified.
         * @param message ListBentoRequest
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ListBentoRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ListBentoRequest to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace ListBentoRequest {

        /** SORTABLE_COLUMN enum. */
        enum SORTABLE_COLUMN {
            created_at = 0,
            name = 1
        }
    }

    /** Properties of a ListBentoResponse. */
    interface IListBentoResponse {

        /** ListBentoResponse status */
        status?: (bentoml.IStatus|null);

        /** ListBentoResponse bentos */
        bentos?: (bentoml.IBento[]|null);
    }

    /** Represents a ListBentoResponse. */
    class ListBentoResponse implements IListBentoResponse {

        /**
         * Constructs a new ListBentoResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IListBentoResponse);

        /** ListBentoResponse status. */
        public status?: (bentoml.IStatus|null);

        /** ListBentoResponse bentos. */
        public bentos: bentoml.IBento[];

        /**
         * Creates a new ListBentoResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns ListBentoResponse instance
         */
        public static create(properties?: bentoml.IListBentoResponse): bentoml.ListBentoResponse;

        /**
         * Encodes the specified ListBentoResponse message. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param message ListBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IListBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListBentoResponse message, length delimited. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param message ListBentoResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IListBentoResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListBentoResponse;

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListBentoResponse;

        /**
         * Verifies a ListBentoResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a ListBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns ListBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListBentoResponse;

        /**
         * Creates a plain object from a ListBentoResponse message. Also converts values to other types if specified.
         * @param message ListBentoResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.ListBentoResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this ListBentoResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Represents a Yatai */
    class Yatai extends $protobuf.rpc.Service {

        /**
         * Constructs a new Yatai service.
         * @param rpcImpl RPC implementation
         * @param [requestDelimited=false] Whether requests are length-delimited
         * @param [responseDelimited=false] Whether responses are length-delimited
         */
        constructor(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean);

        /**
         * Creates new Yatai service using the specified rpc implementation.
         * @param rpcImpl RPC implementation
         * @param [requestDelimited=false] Whether requests are length-delimited
         * @param [responseDelimited=false] Whether responses are length-delimited
         * @returns RPC service. Useful where requests and/or responses are streamed.
         */
        public static create(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean): Yatai;

        /**
         * Calls HealthCheck.
         * @param request Empty message or plain object
         * @param callback Node-style callback called with the error, if any, and HealthCheckResponse
         */
        public healthCheck(request: google.protobuf.IEmpty, callback: bentoml.Yatai.HealthCheckCallback): void;

        /**
         * Calls HealthCheck.
         * @param request Empty message or plain object
         * @returns Promise
         */
        public healthCheck(request: google.protobuf.IEmpty): Promise<bentoml.HealthCheckResponse>;

        /**
         * Calls GetYataiServiceVersion.
         * @param request Empty message or plain object
         * @param callback Node-style callback called with the error, if any, and GetYataiServiceVersionResponse
         */
        public getYataiServiceVersion(request: google.protobuf.IEmpty, callback: bentoml.Yatai.GetYataiServiceVersionCallback): void;

        /**
         * Calls GetYataiServiceVersion.
         * @param request Empty message or plain object
         * @returns Promise
         */
        public getYataiServiceVersion(request: google.protobuf.IEmpty): Promise<bentoml.GetYataiServiceVersionResponse>;

        /**
         * Calls ApplyDeployment.
         * @param request ApplyDeploymentRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and ApplyDeploymentResponse
         */
        public applyDeployment(request: bentoml.IApplyDeploymentRequest, callback: bentoml.Yatai.ApplyDeploymentCallback): void;

        /**
         * Calls ApplyDeployment.
         * @param request ApplyDeploymentRequest message or plain object
         * @returns Promise
         */
        public applyDeployment(request: bentoml.IApplyDeploymentRequest): Promise<bentoml.ApplyDeploymentResponse>;

        /**
         * Calls DeleteDeployment.
         * @param request DeleteDeploymentRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and DeleteDeploymentResponse
         */
        public deleteDeployment(request: bentoml.IDeleteDeploymentRequest, callback: bentoml.Yatai.DeleteDeploymentCallback): void;

        /**
         * Calls DeleteDeployment.
         * @param request DeleteDeploymentRequest message or plain object
         * @returns Promise
         */
        public deleteDeployment(request: bentoml.IDeleteDeploymentRequest): Promise<bentoml.DeleteDeploymentResponse>;

        /**
         * Calls GetDeployment.
         * @param request GetDeploymentRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and GetDeploymentResponse
         */
        public getDeployment(request: bentoml.IGetDeploymentRequest, callback: bentoml.Yatai.GetDeploymentCallback): void;

        /**
         * Calls GetDeployment.
         * @param request GetDeploymentRequest message or plain object
         * @returns Promise
         */
        public getDeployment(request: bentoml.IGetDeploymentRequest): Promise<bentoml.GetDeploymentResponse>;

        /**
         * Calls DescribeDeployment.
         * @param request DescribeDeploymentRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and DescribeDeploymentResponse
         */
        public describeDeployment(request: bentoml.IDescribeDeploymentRequest, callback: bentoml.Yatai.DescribeDeploymentCallback): void;

        /**
         * Calls DescribeDeployment.
         * @param request DescribeDeploymentRequest message or plain object
         * @returns Promise
         */
        public describeDeployment(request: bentoml.IDescribeDeploymentRequest): Promise<bentoml.DescribeDeploymentResponse>;

        /**
         * Calls ListDeployments.
         * @param request ListDeploymentsRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and ListDeploymentsResponse
         */
        public listDeployments(request: bentoml.IListDeploymentsRequest, callback: bentoml.Yatai.ListDeploymentsCallback): void;

        /**
         * Calls ListDeployments.
         * @param request ListDeploymentsRequest message or plain object
         * @returns Promise
         */
        public listDeployments(request: bentoml.IListDeploymentsRequest): Promise<bentoml.ListDeploymentsResponse>;

        /**
         * Calls AddBento.
         * @param request AddBentoRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and AddBentoResponse
         */
        public addBento(request: bentoml.IAddBentoRequest, callback: bentoml.Yatai.AddBentoCallback): void;

        /**
         * Calls AddBento.
         * @param request AddBentoRequest message or plain object
         * @returns Promise
         */
        public addBento(request: bentoml.IAddBentoRequest): Promise<bentoml.AddBentoResponse>;

        /**
         * Calls UpdateBento.
         * @param request UpdateBentoRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and UpdateBentoResponse
         */
        public updateBento(request: bentoml.IUpdateBentoRequest, callback: bentoml.Yatai.UpdateBentoCallback): void;

        /**
         * Calls UpdateBento.
         * @param request UpdateBentoRequest message or plain object
         * @returns Promise
         */
        public updateBento(request: bentoml.IUpdateBentoRequest): Promise<bentoml.UpdateBentoResponse>;

        /**
         * Calls GetBento.
         * @param request GetBentoRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and GetBentoResponse
         */
        public getBento(request: bentoml.IGetBentoRequest, callback: bentoml.Yatai.GetBentoCallback): void;

        /**
         * Calls GetBento.
         * @param request GetBentoRequest message or plain object
         * @returns Promise
         */
        public getBento(request: bentoml.IGetBentoRequest): Promise<bentoml.GetBentoResponse>;

        /**
         * Calls DangerouslyDeleteBento.
         * @param request DangerouslyDeleteBentoRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and DangerouslyDeleteBentoResponse
         */
        public dangerouslyDeleteBento(request: bentoml.IDangerouslyDeleteBentoRequest, callback: bentoml.Yatai.DangerouslyDeleteBentoCallback): void;

        /**
         * Calls DangerouslyDeleteBento.
         * @param request DangerouslyDeleteBentoRequest message or plain object
         * @returns Promise
         */
        public dangerouslyDeleteBento(request: bentoml.IDangerouslyDeleteBentoRequest): Promise<bentoml.DangerouslyDeleteBentoResponse>;

        /**
         * Calls ListBento.
         * @param request ListBentoRequest message or plain object
         * @param callback Node-style callback called with the error, if any, and ListBentoResponse
         */
        public listBento(request: bentoml.IListBentoRequest, callback: bentoml.Yatai.ListBentoCallback): void;

        /**
         * Calls ListBento.
         * @param request ListBentoRequest message or plain object
         * @returns Promise
         */
        public listBento(request: bentoml.IListBentoRequest): Promise<bentoml.ListBentoResponse>;
    }

    namespace Yatai {

        /**
         * Callback as used by {@link bentoml.Yatai#healthCheck}.
         * @param error Error, if any
         * @param [response] HealthCheckResponse
         */
        type HealthCheckCallback = (error: (Error|null), response?: bentoml.HealthCheckResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#getYataiServiceVersion}.
         * @param error Error, if any
         * @param [response] GetYataiServiceVersionResponse
         */
        type GetYataiServiceVersionCallback = (error: (Error|null), response?: bentoml.GetYataiServiceVersionResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#applyDeployment}.
         * @param error Error, if any
         * @param [response] ApplyDeploymentResponse
         */
        type ApplyDeploymentCallback = (error: (Error|null), response?: bentoml.ApplyDeploymentResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#deleteDeployment}.
         * @param error Error, if any
         * @param [response] DeleteDeploymentResponse
         */
        type DeleteDeploymentCallback = (error: (Error|null), response?: bentoml.DeleteDeploymentResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#getDeployment}.
         * @param error Error, if any
         * @param [response] GetDeploymentResponse
         */
        type GetDeploymentCallback = (error: (Error|null), response?: bentoml.GetDeploymentResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#describeDeployment}.
         * @param error Error, if any
         * @param [response] DescribeDeploymentResponse
         */
        type DescribeDeploymentCallback = (error: (Error|null), response?: bentoml.DescribeDeploymentResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#listDeployments}.
         * @param error Error, if any
         * @param [response] ListDeploymentsResponse
         */
        type ListDeploymentsCallback = (error: (Error|null), response?: bentoml.ListDeploymentsResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#addBento}.
         * @param error Error, if any
         * @param [response] AddBentoResponse
         */
        type AddBentoCallback = (error: (Error|null), response?: bentoml.AddBentoResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#updateBento}.
         * @param error Error, if any
         * @param [response] UpdateBentoResponse
         */
        type UpdateBentoCallback = (error: (Error|null), response?: bentoml.UpdateBentoResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#getBento}.
         * @param error Error, if any
         * @param [response] GetBentoResponse
         */
        type GetBentoCallback = (error: (Error|null), response?: bentoml.GetBentoResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#dangerouslyDeleteBento}.
         * @param error Error, if any
         * @param [response] DangerouslyDeleteBentoResponse
         */
        type DangerouslyDeleteBentoCallback = (error: (Error|null), response?: bentoml.DangerouslyDeleteBentoResponse) => void;

        /**
         * Callback as used by {@link bentoml.Yatai#listBento}.
         * @param error Error, if any
         * @param [response] ListBentoResponse
         */
        type ListBentoCallback = (error: (Error|null), response?: bentoml.ListBentoResponse) => void;
    }

    /** Properties of a HealthCheckResponse. */
    interface IHealthCheckResponse {

        /** HealthCheckResponse status */
        status?: (bentoml.IStatus|null);
    }

    /** Represents a HealthCheckResponse. */
    class HealthCheckResponse implements IHealthCheckResponse {

        /**
         * Constructs a new HealthCheckResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IHealthCheckResponse);

        /** HealthCheckResponse status. */
        public status?: (bentoml.IStatus|null);

        /**
         * Creates a new HealthCheckResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns HealthCheckResponse instance
         */
        public static create(properties?: bentoml.IHealthCheckResponse): bentoml.HealthCheckResponse;

        /**
         * Encodes the specified HealthCheckResponse message. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param message HealthCheckResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IHealthCheckResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified HealthCheckResponse message, length delimited. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param message HealthCheckResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IHealthCheckResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.HealthCheckResponse;

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.HealthCheckResponse;

        /**
         * Verifies a HealthCheckResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns HealthCheckResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.HealthCheckResponse;

        /**
         * Creates a plain object from a HealthCheckResponse message. Also converts values to other types if specified.
         * @param message HealthCheckResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.HealthCheckResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this HealthCheckResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GetYataiServiceVersionResponse. */
    interface IGetYataiServiceVersionResponse {

        /** GetYataiServiceVersionResponse status */
        status?: (bentoml.IStatus|null);

        /** GetYataiServiceVersionResponse version */
        version?: (string|null);
    }

    /** Represents a GetYataiServiceVersionResponse. */
    class GetYataiServiceVersionResponse implements IGetYataiServiceVersionResponse {

        /**
         * Constructs a new GetYataiServiceVersionResponse.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IGetYataiServiceVersionResponse);

        /** GetYataiServiceVersionResponse status. */
        public status?: (bentoml.IStatus|null);

        /** GetYataiServiceVersionResponse version. */
        public version: string;

        /**
         * Creates a new GetYataiServiceVersionResponse instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GetYataiServiceVersionResponse instance
         */
        public static create(properties?: bentoml.IGetYataiServiceVersionResponse): bentoml.GetYataiServiceVersionResponse;

        /**
         * Encodes the specified GetYataiServiceVersionResponse message. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param message GetYataiServiceVersionResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IGetYataiServiceVersionResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetYataiServiceVersionResponse message, length delimited. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param message GetYataiServiceVersionResponse message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IGetYataiServiceVersionResponse, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetYataiServiceVersionResponse;

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetYataiServiceVersionResponse;

        /**
         * Verifies a GetYataiServiceVersionResponse message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GetYataiServiceVersionResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetYataiServiceVersionResponse;

        /**
         * Creates a plain object from a GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @param message GetYataiServiceVersionResponse
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.GetYataiServiceVersionResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GetYataiServiceVersionResponse to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a Chunk. */
    interface IChunk {

        /** Chunk content */
        content?: (Uint8Array|null);
    }

    /** Represents a Chunk. */
    class Chunk implements IChunk {

        /**
         * Constructs a new Chunk.
         * @param [properties] Properties to set
         */
        constructor(properties?: bentoml.IChunk);

        /** Chunk content. */
        public content: Uint8Array;

        /**
         * Creates a new Chunk instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Chunk instance
         */
        public static create(properties?: bentoml.IChunk): bentoml.Chunk;

        /**
         * Encodes the specified Chunk message. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param message Chunk message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: bentoml.IChunk, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Chunk message, length delimited. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param message Chunk message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: bentoml.IChunk, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Chunk message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Chunk;

        /**
         * Decodes a Chunk message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Chunk;

        /**
         * Verifies a Chunk message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Chunk
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Chunk;

        /**
         * Creates a plain object from a Chunk message. Also converts values to other types if specified.
         * @param message Chunk
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: bentoml.Chunk, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Chunk to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }
}

/** Namespace google. */
export namespace google {

    /** Namespace protobuf. */
    namespace protobuf {

        /** Properties of a Struct. */
        interface IStruct {

            /** Struct fields */
            fields?: ({ [k: string]: google.protobuf.IValue }|null);
        }

        /** Represents a Struct. */
        class Struct implements IStruct {

            /**
             * Constructs a new Struct.
             * @param [properties] Properties to set
             */
            constructor(properties?: google.protobuf.IStruct);

            /** Struct fields. */
            public fields: { [k: string]: google.protobuf.IValue };

            /**
             * Creates a new Struct instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Struct instance
             */
            public static create(properties?: google.protobuf.IStruct): google.protobuf.Struct;

            /**
             * Encodes the specified Struct message. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param message Struct message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: google.protobuf.IStruct, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Struct message, length delimited. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param message Struct message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: google.protobuf.IStruct, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Struct message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Struct;

            /**
             * Decodes a Struct message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Struct;

            /**
             * Verifies a Struct message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Struct message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Struct
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Struct;

            /**
             * Creates a plain object from a Struct message. Also converts values to other types if specified.
             * @param message Struct
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: google.protobuf.Struct, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Struct to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a Value. */
        interface IValue {

            /** Value nullValue */
            nullValue?: (google.protobuf.NullValue|null);

            /** Value numberValue */
            numberValue?: (number|null);

            /** Value stringValue */
            stringValue?: (string|null);

            /** Value boolValue */
            boolValue?: (boolean|null);

            /** Value structValue */
            structValue?: (google.protobuf.IStruct|null);

            /** Value listValue */
            listValue?: (google.protobuf.IListValue|null);
        }

        /** Represents a Value. */
        class Value implements IValue {

            /**
             * Constructs a new Value.
             * @param [properties] Properties to set
             */
            constructor(properties?: google.protobuf.IValue);

            /** Value nullValue. */
            public nullValue: google.protobuf.NullValue;

            /** Value numberValue. */
            public numberValue: number;

            /** Value stringValue. */
            public stringValue: string;

            /** Value boolValue. */
            public boolValue: boolean;

            /** Value structValue. */
            public structValue?: (google.protobuf.IStruct|null);

            /** Value listValue. */
            public listValue?: (google.protobuf.IListValue|null);

            /** Value kind. */
            public kind?: ("nullValue"|"numberValue"|"stringValue"|"boolValue"|"structValue"|"listValue");

            /**
             * Creates a new Value instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Value instance
             */
            public static create(properties?: google.protobuf.IValue): google.protobuf.Value;

            /**
             * Encodes the specified Value message. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param message Value message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: google.protobuf.IValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Value message, length delimited. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param message Value message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: google.protobuf.IValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Value message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Value;

            /**
             * Decodes a Value message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Value;

            /**
             * Verifies a Value message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Value
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Value;

            /**
             * Creates a plain object from a Value message. Also converts values to other types if specified.
             * @param message Value
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: google.protobuf.Value, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Value to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** NullValue enum. */
        enum NullValue {
            NULL_VALUE = 0
        }

        /** Properties of a ListValue. */
        interface IListValue {

            /** ListValue values */
            values?: (google.protobuf.IValue[]|null);
        }

        /** Represents a ListValue. */
        class ListValue implements IListValue {

            /**
             * Constructs a new ListValue.
             * @param [properties] Properties to set
             */
            constructor(properties?: google.protobuf.IListValue);

            /** ListValue values. */
            public values: google.protobuf.IValue[];

            /**
             * Creates a new ListValue instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ListValue instance
             */
            public static create(properties?: google.protobuf.IListValue): google.protobuf.ListValue;

            /**
             * Encodes the specified ListValue message. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param message ListValue message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: google.protobuf.IListValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ListValue message, length delimited. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param message ListValue message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: google.protobuf.IListValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ListValue message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.ListValue;

            /**
             * Decodes a ListValue message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.ListValue;

            /**
             * Verifies a ListValue message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ListValue
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.ListValue;

            /**
             * Creates a plain object from a ListValue message. Also converts values to other types if specified.
             * @param message ListValue
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: google.protobuf.ListValue, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ListValue to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a Timestamp. */
        interface ITimestamp {

            /** Timestamp seconds */
            seconds?: (number|null);

            /** Timestamp nanos */
            nanos?: (number|null);
        }

        /** Represents a Timestamp. */
        class Timestamp implements ITimestamp {

            /**
             * Constructs a new Timestamp.
             * @param [properties] Properties to set
             */
            constructor(properties?: google.protobuf.ITimestamp);

            /** Timestamp seconds. */
            public seconds: number;

            /** Timestamp nanos. */
            public nanos: number;

            /**
             * Creates a new Timestamp instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Timestamp instance
             */
            public static create(properties?: google.protobuf.ITimestamp): google.protobuf.Timestamp;

            /**
             * Encodes the specified Timestamp message. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param message Timestamp message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: google.protobuf.ITimestamp, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Timestamp message, length delimited. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param message Timestamp message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: google.protobuf.ITimestamp, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Timestamp message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Timestamp;

            /**
             * Decodes a Timestamp message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Timestamp;

            /**
             * Verifies a Timestamp message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Timestamp
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Timestamp;

            /**
             * Creates a plain object from a Timestamp message. Also converts values to other types if specified.
             * @param message Timestamp
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: google.protobuf.Timestamp, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Timestamp to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of an Empty. */
        interface IEmpty {
        }

        /** Represents an Empty. */
        class Empty implements IEmpty {

            /**
             * Constructs a new Empty.
             * @param [properties] Properties to set
             */
            constructor(properties?: google.protobuf.IEmpty);

            /**
             * Creates a new Empty instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Empty instance
             */
            public static create(properties?: google.protobuf.IEmpty): google.protobuf.Empty;

            /**
             * Encodes the specified Empty message. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param message Empty message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: google.protobuf.IEmpty, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Empty message, length delimited. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param message Empty message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: google.protobuf.IEmpty, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an Empty message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Empty;

            /**
             * Decodes an Empty message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Empty;

            /**
             * Verifies an Empty message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Empty
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Empty;

            /**
             * Creates a plain object from an Empty message. Also converts values to other types if specified.
             * @param message Empty
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: google.protobuf.Empty, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Empty to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }
}
