import * as $protobuf from "protobufjs";

/**
 * Namespace bentoml.
 * @exports bentoml
 * @namespace
 */
export namespace bentoml {

    type DeploymentSpec$Properties = {
        bento_name?: string;
        bento_version?: string;
        operator?: bentoml.DeploymentSpec.DeploymentOperator;
        custom_operator_config?: bentoml.DeploymentSpec.CustomOperatorConfig$Properties;
        sagemaker_operator_config?: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties;
        aws_lambda_operator_config?: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties;
        azure_functions_operator_config?: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties;
    };

    /**
     * Constructs a new DeploymentSpec.
     * @exports bentoml.DeploymentSpec
     * @constructor
     * @param {bentoml.DeploymentSpec$Properties=} [properties] Properties to set
     */
    class DeploymentSpec {

        /**
         * Constructs a new DeploymentSpec.
         * @exports bentoml.DeploymentSpec
         * @constructor
         * @param {bentoml.DeploymentSpec$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DeploymentSpec$Properties);

        /**
         * DeploymentSpec bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * DeploymentSpec bento_version.
         * @type {string|undefined}
         */
        public bento_version?: string;

        /**
         * DeploymentSpec operator.
         * @type {bentoml.DeploymentSpec.DeploymentOperator|undefined}
         */
        public operator?: bentoml.DeploymentSpec.DeploymentOperator;

        /**
         * DeploymentSpec custom_operator_config.
         * @type {bentoml.DeploymentSpec.CustomOperatorConfig$Properties|undefined}
         */
        public custom_operator_config?: bentoml.DeploymentSpec.CustomOperatorConfig$Properties;

        /**
         * DeploymentSpec sagemaker_operator_config.
         * @type {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties|undefined}
         */
        public sagemaker_operator_config?: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties;

        /**
         * DeploymentSpec aws_lambda_operator_config.
         * @type {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties|undefined}
         */
        public aws_lambda_operator_config?: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties;

        /**
         * DeploymentSpec azure_functions_operator_config.
         * @type {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties|undefined}
         */
        public azure_functions_operator_config?: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties;

        /**
         * DeploymentSpec deployment_operator_config.
         * @name bentoml.DeploymentSpec#deployment_operator_config
         * @type {string|undefined}
         */
        public deployment_operator_config?: string;

        /**
         * Creates a new DeploymentSpec instance using the specified properties.
         * @param {bentoml.DeploymentSpec$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentSpec} DeploymentSpec instance
         */
        public static create(properties?: bentoml.DeploymentSpec$Properties): bentoml.DeploymentSpec;

        /**
         * Encodes the specified DeploymentSpec message. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param {bentoml.DeploymentSpec$Properties} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DeploymentSpec$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentSpec message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param {bentoml.DeploymentSpec$Properties} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DeploymentSpec$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec;

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec;

        /**
         * Verifies a DeploymentSpec message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec;

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentSpec.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         */
        public static from(object: { [k: string]: any }): bentoml.DeploymentSpec;

        /**
         * Creates a plain object from a DeploymentSpec message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentSpec} message DeploymentSpec
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DeploymentSpec, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DeploymentSpec message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentSpec to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace DeploymentSpec {

        /**
         * DeploymentOperator enum.
         * @name DeploymentOperator
         * @memberof bentoml.DeploymentSpec
         * @enum {number}
         * @property {number} UNSET=0 UNSET value
         * @property {number} CUSTOM=1 CUSTOM value
         * @property {number} AWS_SAGEMAKER=2 AWS_SAGEMAKER value
         * @property {number} AWS_LAMBDA=3 AWS_LAMBDA value
         * @property {number} AZURE_FUNCTIONS=4 AZURE_FUNCTIONS value
         */
        enum DeploymentOperator {
            UNSET = 0,
            CUSTOM = 1,
            AWS_SAGEMAKER = 2,
            AWS_LAMBDA = 3,
            AZURE_FUNCTIONS = 4
        }

        type CustomOperatorConfig$Properties = {
            name?: string;
            config?: google.protobuf.Struct$Properties;
        };

        /**
         * Constructs a new CustomOperatorConfig.
         * @exports bentoml.DeploymentSpec.CustomOperatorConfig
         * @constructor
         * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties=} [properties] Properties to set
         */
        class CustomOperatorConfig {

            /**
             * Constructs a new CustomOperatorConfig.
             * @exports bentoml.DeploymentSpec.CustomOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.CustomOperatorConfig$Properties);

            /**
             * CustomOperatorConfig name.
             * @type {string|undefined}
             */
            public name?: string;

            /**
             * CustomOperatorConfig config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            public config?: google.protobuf.Struct$Properties;

            /**
             * Creates a new CustomOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.CustomOperatorConfig$Properties): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Encodes the specified CustomOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.DeploymentSpec.CustomOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified CustomOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.CustomOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Verifies a CustomOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.CustomOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             */
            public static from(object: { [k: string]: any }): bentoml.DeploymentSpec.CustomOperatorConfig;

            /**
             * Creates a plain object from a CustomOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig} message CustomOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.CustomOperatorConfig, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this CustomOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this CustomOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type SageMakerOperatorConfig$Properties = {
            region?: string;
            instance_type?: string;
            instance_count?: number;
            api_name?: string;
            num_of_gunicorn_workers_per_instance?: number;
            timeout?: number;
        };

        /**
         * Constructs a new SageMakerOperatorConfig.
         * @exports bentoml.DeploymentSpec.SageMakerOperatorConfig
         * @constructor
         * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties=} [properties] Properties to set
         */
        class SageMakerOperatorConfig {

            /**
             * Constructs a new SageMakerOperatorConfig.
             * @exports bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties);

            /**
             * SageMakerOperatorConfig region.
             * @type {string|undefined}
             */
            public region?: string;

            /**
             * SageMakerOperatorConfig instance_type.
             * @type {string|undefined}
             */
            public instance_type?: string;

            /**
             * SageMakerOperatorConfig instance_count.
             * @type {number|undefined}
             */
            public instance_count?: number;

            /**
             * SageMakerOperatorConfig api_name.
             * @type {string|undefined}
             */
            public api_name?: string;

            /**
             * SageMakerOperatorConfig num_of_gunicorn_workers_per_instance.
             * @type {number|undefined}
             */
            public num_of_gunicorn_workers_per_instance?: number;

            /**
             * SageMakerOperatorConfig timeout.
             * @type {number|undefined}
             */
            public timeout?: number;

            /**
             * Creates a new SageMakerOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Encodes the specified SageMakerOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified SageMakerOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Verifies a SageMakerOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             */
            public static from(object: { [k: string]: any }): bentoml.DeploymentSpec.SageMakerOperatorConfig;

            /**
             * Creates a plain object from a SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig} message SageMakerOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.SageMakerOperatorConfig, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this SageMakerOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type AwsLambdaOperatorConfig$Properties = {
            region?: string;
            api_name?: string;
            memory_size?: number;
            timeout?: number;
        };

        /**
         * Constructs a new AwsLambdaOperatorConfig.
         * @exports bentoml.DeploymentSpec.AwsLambdaOperatorConfig
         * @constructor
         * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties=} [properties] Properties to set
         */
        class AwsLambdaOperatorConfig {

            /**
             * Constructs a new AwsLambdaOperatorConfig.
             * @exports bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties);

            /**
             * AwsLambdaOperatorConfig region.
             * @type {string|undefined}
             */
            public region?: string;

            /**
             * AwsLambdaOperatorConfig api_name.
             * @type {string|undefined}
             */
            public api_name?: string;

            /**
             * AwsLambdaOperatorConfig memory_size.
             * @type {number|undefined}
             */
            public memory_size?: number;

            /**
             * AwsLambdaOperatorConfig timeout.
             * @type {number|undefined}
             */
            public timeout?: number;

            /**
             * Creates a new AwsLambdaOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Encodes the specified AwsLambdaOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AwsLambdaOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Verifies an AwsLambdaOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             */
            public static from(object: { [k: string]: any }): bentoml.DeploymentSpec.AwsLambdaOperatorConfig;

            /**
             * Creates a plain object from an AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} message AwsLambdaOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.AwsLambdaOperatorConfig, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this AwsLambdaOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type AzureFunctionsOperatorConfig$Properties = {
            location?: string;
            premium_plan_sku?: string;
            min_instances?: number;
            max_burst?: number;
            function_auth_level?: string;
        };

        /**
         * Constructs a new AzureFunctionsOperatorConfig.
         * @exports bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
         * @constructor
         * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties=} [properties] Properties to set
         */
        class AzureFunctionsOperatorConfig {

            /**
             * Constructs a new AzureFunctionsOperatorConfig.
             * @exports bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties);

            /**
             * AzureFunctionsOperatorConfig location.
             * @type {string|undefined}
             */
            public location?: string;

            /**
             * AzureFunctionsOperatorConfig premium_plan_sku.
             * @type {string|undefined}
             */
            public premium_plan_sku?: string;

            /**
             * AzureFunctionsOperatorConfig min_instances.
             * @type {number|undefined}
             */
            public min_instances?: number;

            /**
             * AzureFunctionsOperatorConfig max_burst.
             * @type {number|undefined}
             */
            public max_burst?: number;

            /**
             * AzureFunctionsOperatorConfig function_auth_level.
             * @type {string|undefined}
             */
            public function_auth_level?: string;

            /**
             * Creates a new AzureFunctionsOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig instance
             */
            public static create(properties?: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Verifies an AzureFunctionsOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             */
            public static fromObject(object: { [k: string]: any }): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             */
            public static from(object: { [k: string]: any }): bentoml.DeploymentSpec.AzureFunctionsOperatorConfig;

            /**
             * Creates a plain object from an AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} message AzureFunctionsOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.DeploymentSpec.AzureFunctionsOperatorConfig, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this AzureFunctionsOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    type DeploymentState$Properties = {
        state?: bentoml.DeploymentState.State;
        error_message?: string;
        info_json?: string;
        timestamp?: google.protobuf.Timestamp$Properties;
    };

    /**
     * Constructs a new DeploymentState.
     * @exports bentoml.DeploymentState
     * @constructor
     * @param {bentoml.DeploymentState$Properties=} [properties] Properties to set
     */
    class DeploymentState {

        /**
         * Constructs a new DeploymentState.
         * @exports bentoml.DeploymentState
         * @constructor
         * @param {bentoml.DeploymentState$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DeploymentState$Properties);

        /**
         * DeploymentState state.
         * @type {bentoml.DeploymentState.State|undefined}
         */
        public state?: bentoml.DeploymentState.State;

        /**
         * DeploymentState error_message.
         * @type {string|undefined}
         */
        public error_message?: string;

        /**
         * DeploymentState info_json.
         * @type {string|undefined}
         */
        public info_json?: string;

        /**
         * DeploymentState timestamp.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        public timestamp?: google.protobuf.Timestamp$Properties;

        /**
         * Creates a new DeploymentState instance using the specified properties.
         * @param {bentoml.DeploymentState$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentState} DeploymentState instance
         */
        public static create(properties?: bentoml.DeploymentState$Properties): bentoml.DeploymentState;

        /**
         * Encodes the specified DeploymentState message. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param {bentoml.DeploymentState$Properties} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DeploymentState$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentState message, length delimited. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param {bentoml.DeploymentState$Properties} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DeploymentState$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentState message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentState} DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentState;

        /**
         * Decodes a DeploymentState message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentState} DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentState;

        /**
         * Verifies a DeploymentState message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentState} DeploymentState
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentState;

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentState.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentState} DeploymentState
         */
        public static from(object: { [k: string]: any }): bentoml.DeploymentState;

        /**
         * Creates a plain object from a DeploymentState message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentState} message DeploymentState
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DeploymentState, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DeploymentState message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentState to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace DeploymentState {

        /**
         * State enum.
         * @name State
         * @memberof bentoml.DeploymentState
         * @enum {number}
         * @property {number} PENDING=0 PENDING value
         * @property {number} RUNNING=1 RUNNING value
         * @property {number} SUCCEEDED=2 SUCCEEDED value
         * @property {number} FAILED=3 FAILED value
         * @property {number} UNKNOWN=4 UNKNOWN value
         * @property {number} COMPLETED=5 COMPLETED value
         * @property {number} CRASH_LOOP_BACK_OFF=6 CRASH_LOOP_BACK_OFF value
         * @property {number} ERROR=7 ERROR value
         * @property {number} INACTIVATED=8 INACTIVATED value
         */
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

    type Deployment$Properties = {
        namespace?: string;
        name?: string;
        spec?: bentoml.DeploymentSpec$Properties;
        state?: bentoml.DeploymentState$Properties;
        annotations?: { [k: string]: string };
        labels?: { [k: string]: string };
        created_at?: google.protobuf.Timestamp$Properties;
        last_updated_at?: google.protobuf.Timestamp$Properties;
    };

    /**
     * Constructs a new Deployment.
     * @exports bentoml.Deployment
     * @constructor
     * @param {bentoml.Deployment$Properties=} [properties] Properties to set
     */
    class Deployment {

        /**
         * Constructs a new Deployment.
         * @exports bentoml.Deployment
         * @constructor
         * @param {bentoml.Deployment$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.Deployment$Properties);

        /**
         * Deployment namespace.
         * @type {string|undefined}
         */
        public namespace?: string;

        /**
         * Deployment name.
         * @type {string|undefined}
         */
        public name?: string;

        /**
         * Deployment spec.
         * @type {bentoml.DeploymentSpec$Properties|undefined}
         */
        public spec?: bentoml.DeploymentSpec$Properties;

        /**
         * Deployment state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        public state?: bentoml.DeploymentState$Properties;

        /**
         * Deployment annotations.
         * @type {Object.<string,string>|undefined}
         */
        public annotations?: { [k: string]: string };

        /**
         * Deployment labels.
         * @type {Object.<string,string>|undefined}
         */
        public labels?: { [k: string]: string };

        /**
         * Deployment created_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        public created_at?: google.protobuf.Timestamp$Properties;

        /**
         * Deployment last_updated_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        public last_updated_at?: google.protobuf.Timestamp$Properties;

        /**
         * Creates a new Deployment instance using the specified properties.
         * @param {bentoml.Deployment$Properties=} [properties] Properties to set
         * @returns {bentoml.Deployment} Deployment instance
         */
        public static create(properties?: bentoml.Deployment$Properties): bentoml.Deployment;

        /**
         * Encodes the specified Deployment message. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param {bentoml.Deployment$Properties} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.Deployment$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Deployment message, length delimited. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param {bentoml.Deployment$Properties} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.Deployment$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Deployment message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Deployment} Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Deployment;

        /**
         * Decodes a Deployment message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Deployment} Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Deployment;

        /**
         * Verifies a Deployment message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Deployment} Deployment
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Deployment;

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Deployment.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Deployment} Deployment
         */
        public static from(object: { [k: string]: any }): bentoml.Deployment;

        /**
         * Creates a plain object from a Deployment message. Also converts values to other types if specified.
         * @param {bentoml.Deployment} message Deployment
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.Deployment, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this Deployment message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this Deployment to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DeploymentStatus$Properties = {
        state?: bentoml.DeploymentState$Properties;
    };

    /**
     * Constructs a new DeploymentStatus.
     * @exports bentoml.DeploymentStatus
     * @constructor
     * @param {bentoml.DeploymentStatus$Properties=} [properties] Properties to set
     */
    class DeploymentStatus {

        /**
         * Constructs a new DeploymentStatus.
         * @exports bentoml.DeploymentStatus
         * @constructor
         * @param {bentoml.DeploymentStatus$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DeploymentStatus$Properties);

        /**
         * DeploymentStatus state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        public state?: bentoml.DeploymentState$Properties;

        /**
         * Creates a new DeploymentStatus instance using the specified properties.
         * @param {bentoml.DeploymentStatus$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentStatus} DeploymentStatus instance
         */
        public static create(properties?: bentoml.DeploymentStatus$Properties): bentoml.DeploymentStatus;

        /**
         * Encodes the specified DeploymentStatus message. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param {bentoml.DeploymentStatus$Properties} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DeploymentStatus$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeploymentStatus message, length delimited. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param {bentoml.DeploymentStatus$Properties} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DeploymentStatus$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeploymentStatus;

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeploymentStatus;

        /**
         * Verifies a DeploymentStatus message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeploymentStatus;

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentStatus.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         */
        public static from(object: { [k: string]: any }): bentoml.DeploymentStatus;

        /**
         * Creates a plain object from a DeploymentStatus message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentStatus} message DeploymentStatus
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DeploymentStatus, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DeploymentStatus message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DeploymentStatus to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type ApplyDeploymentRequest$Properties = {
        deployment?: bentoml.Deployment$Properties;
    };

    /**
     * Constructs a new ApplyDeploymentRequest.
     * @exports bentoml.ApplyDeploymentRequest
     * @constructor
     * @param {bentoml.ApplyDeploymentRequest$Properties=} [properties] Properties to set
     */
    class ApplyDeploymentRequest {

        /**
         * Constructs a new ApplyDeploymentRequest.
         * @exports bentoml.ApplyDeploymentRequest
         * @constructor
         * @param {bentoml.ApplyDeploymentRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ApplyDeploymentRequest$Properties);

        /**
         * ApplyDeploymentRequest deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        public deployment?: bentoml.Deployment$Properties;

        /**
         * Creates a new ApplyDeploymentRequest instance using the specified properties.
         * @param {bentoml.ApplyDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest instance
         */
        public static create(properties?: bentoml.ApplyDeploymentRequest$Properties): bentoml.ApplyDeploymentRequest;

        /**
         * Encodes the specified ApplyDeploymentRequest message. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentRequest$Properties} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ApplyDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ApplyDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentRequest$Properties} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ApplyDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ApplyDeploymentRequest;

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ApplyDeploymentRequest;

        /**
         * Verifies an ApplyDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ApplyDeploymentRequest;

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ApplyDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         */
        public static from(object: { [k: string]: any }): bentoml.ApplyDeploymentRequest;

        /**
         * Creates a plain object from an ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.ApplyDeploymentRequest} message ApplyDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ApplyDeploymentRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ApplyDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type ApplyDeploymentResponse$Properties = {
        status?: bentoml.Status$Properties;
        deployment?: bentoml.Deployment$Properties;
    };

    /**
     * Constructs a new ApplyDeploymentResponse.
     * @exports bentoml.ApplyDeploymentResponse
     * @constructor
     * @param {bentoml.ApplyDeploymentResponse$Properties=} [properties] Properties to set
     */
    class ApplyDeploymentResponse {

        /**
         * Constructs a new ApplyDeploymentResponse.
         * @exports bentoml.ApplyDeploymentResponse
         * @constructor
         * @param {bentoml.ApplyDeploymentResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ApplyDeploymentResponse$Properties);

        /**
         * ApplyDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * ApplyDeploymentResponse deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        public deployment?: bentoml.Deployment$Properties;

        /**
         * Creates a new ApplyDeploymentResponse instance using the specified properties.
         * @param {bentoml.ApplyDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse instance
         */
        public static create(properties?: bentoml.ApplyDeploymentResponse$Properties): bentoml.ApplyDeploymentResponse;

        /**
         * Encodes the specified ApplyDeploymentResponse message. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentResponse$Properties} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ApplyDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ApplyDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentResponse$Properties} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ApplyDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ApplyDeploymentResponse;

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ApplyDeploymentResponse;

        /**
         * Verifies an ApplyDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ApplyDeploymentResponse;

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ApplyDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         */
        public static from(object: { [k: string]: any }): bentoml.ApplyDeploymentResponse;

        /**
         * Creates a plain object from an ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.ApplyDeploymentResponse} message ApplyDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ApplyDeploymentResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ApplyDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DeleteDeploymentRequest$Properties = {
        deployment_name?: string;
        namespace?: string;
        force_delete?: boolean;
    };

    /**
     * Constructs a new DeleteDeploymentRequest.
     * @exports bentoml.DeleteDeploymentRequest
     * @constructor
     * @param {bentoml.DeleteDeploymentRequest$Properties=} [properties] Properties to set
     */
    class DeleteDeploymentRequest {

        /**
         * Constructs a new DeleteDeploymentRequest.
         * @exports bentoml.DeleteDeploymentRequest
         * @constructor
         * @param {bentoml.DeleteDeploymentRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DeleteDeploymentRequest$Properties);

        /**
         * DeleteDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        public deployment_name?: string;

        /**
         * DeleteDeploymentRequest namespace.
         * @type {string|undefined}
         */
        public namespace?: string;

        /**
         * DeleteDeploymentRequest force_delete.
         * @type {boolean|undefined}
         */
        public force_delete?: boolean;

        /**
         * Creates a new DeleteDeploymentRequest instance using the specified properties.
         * @param {bentoml.DeleteDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest instance
         */
        public static create(properties?: bentoml.DeleteDeploymentRequest$Properties): bentoml.DeleteDeploymentRequest;

        /**
         * Encodes the specified DeleteDeploymentRequest message. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentRequest$Properties} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DeleteDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeleteDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentRequest$Properties} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DeleteDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeleteDeploymentRequest;

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeleteDeploymentRequest;

        /**
         * Verifies a DeleteDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeleteDeploymentRequest;

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeleteDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         */
        public static from(object: { [k: string]: any }): bentoml.DeleteDeploymentRequest;

        /**
         * Creates a plain object from a DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.DeleteDeploymentRequest} message DeleteDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DeleteDeploymentRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DeleteDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DeleteDeploymentResponse$Properties = {
        status?: bentoml.Status$Properties;
    };

    /**
     * Constructs a new DeleteDeploymentResponse.
     * @exports bentoml.DeleteDeploymentResponse
     * @constructor
     * @param {bentoml.DeleteDeploymentResponse$Properties=} [properties] Properties to set
     */
    class DeleteDeploymentResponse {

        /**
         * Constructs a new DeleteDeploymentResponse.
         * @exports bentoml.DeleteDeploymentResponse
         * @constructor
         * @param {bentoml.DeleteDeploymentResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DeleteDeploymentResponse$Properties);

        /**
         * DeleteDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * Creates a new DeleteDeploymentResponse instance using the specified properties.
         * @param {bentoml.DeleteDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse instance
         */
        public static create(properties?: bentoml.DeleteDeploymentResponse$Properties): bentoml.DeleteDeploymentResponse;

        /**
         * Encodes the specified DeleteDeploymentResponse message. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentResponse$Properties} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DeleteDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DeleteDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentResponse$Properties} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DeleteDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DeleteDeploymentResponse;

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DeleteDeploymentResponse;

        /**
         * Verifies a DeleteDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DeleteDeploymentResponse;

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeleteDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         */
        public static from(object: { [k: string]: any }): bentoml.DeleteDeploymentResponse;

        /**
         * Creates a plain object from a DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.DeleteDeploymentResponse} message DeleteDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DeleteDeploymentResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DeleteDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type GetDeploymentRequest$Properties = {
        deployment_name?: string;
        namespace?: string;
    };

    /**
     * Constructs a new GetDeploymentRequest.
     * @exports bentoml.GetDeploymentRequest
     * @constructor
     * @param {bentoml.GetDeploymentRequest$Properties=} [properties] Properties to set
     */
    class GetDeploymentRequest {

        /**
         * Constructs a new GetDeploymentRequest.
         * @exports bentoml.GetDeploymentRequest
         * @constructor
         * @param {bentoml.GetDeploymentRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.GetDeploymentRequest$Properties);

        /**
         * GetDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        public deployment_name?: string;

        /**
         * GetDeploymentRequest namespace.
         * @type {string|undefined}
         */
        public namespace?: string;

        /**
         * Creates a new GetDeploymentRequest instance using the specified properties.
         * @param {bentoml.GetDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest instance
         */
        public static create(properties?: bentoml.GetDeploymentRequest$Properties): bentoml.GetDeploymentRequest;

        /**
         * Encodes the specified GetDeploymentRequest message. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param {bentoml.GetDeploymentRequest$Properties} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.GetDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param {bentoml.GetDeploymentRequest$Properties} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.GetDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetDeploymentRequest;

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetDeploymentRequest;

        /**
         * Verifies a GetDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetDeploymentRequest;

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         */
        public static from(object: { [k: string]: any }): bentoml.GetDeploymentRequest;

        /**
         * Creates a plain object from a GetDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.GetDeploymentRequest} message GetDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.GetDeploymentRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this GetDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this GetDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type GetDeploymentResponse$Properties = {
        status?: bentoml.Status$Properties;
        deployment?: bentoml.Deployment$Properties;
    };

    /**
     * Constructs a new GetDeploymentResponse.
     * @exports bentoml.GetDeploymentResponse
     * @constructor
     * @param {bentoml.GetDeploymentResponse$Properties=} [properties] Properties to set
     */
    class GetDeploymentResponse {

        /**
         * Constructs a new GetDeploymentResponse.
         * @exports bentoml.GetDeploymentResponse
         * @constructor
         * @param {bentoml.GetDeploymentResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.GetDeploymentResponse$Properties);

        /**
         * GetDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * GetDeploymentResponse deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        public deployment?: bentoml.Deployment$Properties;

        /**
         * Creates a new GetDeploymentResponse instance using the specified properties.
         * @param {bentoml.GetDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse instance
         */
        public static create(properties?: bentoml.GetDeploymentResponse$Properties): bentoml.GetDeploymentResponse;

        /**
         * Encodes the specified GetDeploymentResponse message. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param {bentoml.GetDeploymentResponse$Properties} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.GetDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param {bentoml.GetDeploymentResponse$Properties} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.GetDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetDeploymentResponse;

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetDeploymentResponse;

        /**
         * Verifies a GetDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetDeploymentResponse;

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         */
        public static from(object: { [k: string]: any }): bentoml.GetDeploymentResponse;

        /**
         * Creates a plain object from a GetDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetDeploymentResponse} message GetDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.GetDeploymentResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this GetDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this GetDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DescribeDeploymentRequest$Properties = {
        deployment_name?: string;
        namespace?: string;
    };

    /**
     * Constructs a new DescribeDeploymentRequest.
     * @exports bentoml.DescribeDeploymentRequest
     * @constructor
     * @param {bentoml.DescribeDeploymentRequest$Properties=} [properties] Properties to set
     */
    class DescribeDeploymentRequest {

        /**
         * Constructs a new DescribeDeploymentRequest.
         * @exports bentoml.DescribeDeploymentRequest
         * @constructor
         * @param {bentoml.DescribeDeploymentRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DescribeDeploymentRequest$Properties);

        /**
         * DescribeDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        public deployment_name?: string;

        /**
         * DescribeDeploymentRequest namespace.
         * @type {string|undefined}
         */
        public namespace?: string;

        /**
         * Creates a new DescribeDeploymentRequest instance using the specified properties.
         * @param {bentoml.DescribeDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest instance
         */
        public static create(properties?: bentoml.DescribeDeploymentRequest$Properties): bentoml.DescribeDeploymentRequest;

        /**
         * Encodes the specified DescribeDeploymentRequest message. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentRequest$Properties} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DescribeDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DescribeDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentRequest$Properties} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DescribeDeploymentRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DescribeDeploymentRequest;

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DescribeDeploymentRequest;

        /**
         * Verifies a DescribeDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DescribeDeploymentRequest;

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DescribeDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         */
        public static from(object: { [k: string]: any }): bentoml.DescribeDeploymentRequest;

        /**
         * Creates a plain object from a DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.DescribeDeploymentRequest} message DescribeDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DescribeDeploymentRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DescribeDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DescribeDeploymentResponse$Properties = {
        status?: bentoml.Status$Properties;
        state?: bentoml.DeploymentState$Properties;
    };

    /**
     * Constructs a new DescribeDeploymentResponse.
     * @exports bentoml.DescribeDeploymentResponse
     * @constructor
     * @param {bentoml.DescribeDeploymentResponse$Properties=} [properties] Properties to set
     */
    class DescribeDeploymentResponse {

        /**
         * Constructs a new DescribeDeploymentResponse.
         * @exports bentoml.DescribeDeploymentResponse
         * @constructor
         * @param {bentoml.DescribeDeploymentResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DescribeDeploymentResponse$Properties);

        /**
         * DescribeDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * DescribeDeploymentResponse state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        public state?: bentoml.DeploymentState$Properties;

        /**
         * Creates a new DescribeDeploymentResponse instance using the specified properties.
         * @param {bentoml.DescribeDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse instance
         */
        public static create(properties?: bentoml.DescribeDeploymentResponse$Properties): bentoml.DescribeDeploymentResponse;

        /**
         * Encodes the specified DescribeDeploymentResponse message. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentResponse$Properties} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DescribeDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DescribeDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentResponse$Properties} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DescribeDeploymentResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DescribeDeploymentResponse;

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DescribeDeploymentResponse;

        /**
         * Verifies a DescribeDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DescribeDeploymentResponse;

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DescribeDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         */
        public static from(object: { [k: string]: any }): bentoml.DescribeDeploymentResponse;

        /**
         * Creates a plain object from a DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.DescribeDeploymentResponse} message DescribeDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DescribeDeploymentResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DescribeDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type ListDeploymentsRequest$Properties = {
        namespace?: string;
        offset?: number;
        limit?: number;
        operator?: bentoml.DeploymentSpec.DeploymentOperator;
        order_by?: bentoml.ListDeploymentsRequest.SORTABLE_COLUMN;
        ascending_order?: boolean;
        labels_query?: string;
    };

    /**
     * Constructs a new ListDeploymentsRequest.
     * @exports bentoml.ListDeploymentsRequest
     * @constructor
     * @param {bentoml.ListDeploymentsRequest$Properties=} [properties] Properties to set
     */
    class ListDeploymentsRequest {

        /**
         * Constructs a new ListDeploymentsRequest.
         * @exports bentoml.ListDeploymentsRequest
         * @constructor
         * @param {bentoml.ListDeploymentsRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ListDeploymentsRequest$Properties);

        /**
         * ListDeploymentsRequest namespace.
         * @type {string|undefined}
         */
        public namespace?: string;

        /**
         * ListDeploymentsRequest offset.
         * @type {number|undefined}
         */
        public offset?: number;

        /**
         * ListDeploymentsRequest limit.
         * @type {number|undefined}
         */
        public limit?: number;

        /**
         * ListDeploymentsRequest operator.
         * @type {bentoml.DeploymentSpec.DeploymentOperator|undefined}
         */
        public operator?: bentoml.DeploymentSpec.DeploymentOperator;

        /**
         * ListDeploymentsRequest order_by.
         * @type {bentoml.ListDeploymentsRequest.SORTABLE_COLUMN|undefined}
         */
        public order_by?: bentoml.ListDeploymentsRequest.SORTABLE_COLUMN;

        /**
         * ListDeploymentsRequest ascending_order.
         * @type {boolean|undefined}
         */
        public ascending_order?: boolean;

        /**
         * ListDeploymentsRequest labels_query.
         * @type {string|undefined}
         */
        public labels_query?: string;

        /**
         * Creates a new ListDeploymentsRequest instance using the specified properties.
         * @param {bentoml.ListDeploymentsRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest instance
         */
        public static create(properties?: bentoml.ListDeploymentsRequest$Properties): bentoml.ListDeploymentsRequest;

        /**
         * Encodes the specified ListDeploymentsRequest message. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param {bentoml.ListDeploymentsRequest$Properties} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ListDeploymentsRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListDeploymentsRequest message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param {bentoml.ListDeploymentsRequest$Properties} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ListDeploymentsRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListDeploymentsRequest;

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListDeploymentsRequest;

        /**
         * Verifies a ListDeploymentsRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListDeploymentsRequest;

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListDeploymentsRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         */
        public static from(object: { [k: string]: any }): bentoml.ListDeploymentsRequest;

        /**
         * Creates a plain object from a ListDeploymentsRequest message. Also converts values to other types if specified.
         * @param {bentoml.ListDeploymentsRequest} message ListDeploymentsRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ListDeploymentsRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ListDeploymentsRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ListDeploymentsRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace ListDeploymentsRequest {

        /**
         * SORTABLE_COLUMN enum.
         * @name SORTABLE_COLUMN
         * @memberof bentoml.ListDeploymentsRequest
         * @enum {number}
         * @property {number} created_at=0 created_at value
         * @property {number} name=1 name value
         */
        enum SORTABLE_COLUMN {
            created_at = 0,
            name = 1
        }
    }

    type ListDeploymentsResponse$Properties = {
        status?: bentoml.Status$Properties;
        deployments?: bentoml.Deployment$Properties[];
    };

    /**
     * Constructs a new ListDeploymentsResponse.
     * @exports bentoml.ListDeploymentsResponse
     * @constructor
     * @param {bentoml.ListDeploymentsResponse$Properties=} [properties] Properties to set
     */
    class ListDeploymentsResponse {

        /**
         * Constructs a new ListDeploymentsResponse.
         * @exports bentoml.ListDeploymentsResponse
         * @constructor
         * @param {bentoml.ListDeploymentsResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ListDeploymentsResponse$Properties);

        /**
         * ListDeploymentsResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * ListDeploymentsResponse deployments.
         * @type {Array.<bentoml.Deployment$Properties>|undefined}
         */
        public deployments?: bentoml.Deployment$Properties[];

        /**
         * Creates a new ListDeploymentsResponse instance using the specified properties.
         * @param {bentoml.ListDeploymentsResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse instance
         */
        public static create(properties?: bentoml.ListDeploymentsResponse$Properties): bentoml.ListDeploymentsResponse;

        /**
         * Encodes the specified ListDeploymentsResponse message. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param {bentoml.ListDeploymentsResponse$Properties} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ListDeploymentsResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListDeploymentsResponse message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param {bentoml.ListDeploymentsResponse$Properties} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ListDeploymentsResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListDeploymentsResponse;

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListDeploymentsResponse;

        /**
         * Verifies a ListDeploymentsResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a ListDeploymentsResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListDeploymentsResponse;

        /**
         * Creates a ListDeploymentsResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListDeploymentsResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         */
        public static from(object: { [k: string]: any }): bentoml.ListDeploymentsResponse;

        /**
         * Creates a plain object from a ListDeploymentsResponse message. Also converts values to other types if specified.
         * @param {bentoml.ListDeploymentsResponse} message ListDeploymentsResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ListDeploymentsResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ListDeploymentsResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ListDeploymentsResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type Status$Properties = {
        status_code?: bentoml.Status.Code;
        error_message?: string;
    };

    /**
     * Constructs a new Status.
     * @exports bentoml.Status
     * @constructor
     * @param {bentoml.Status$Properties=} [properties] Properties to set
     */
    class Status {

        /**
         * Constructs a new Status.
         * @exports bentoml.Status
         * @constructor
         * @param {bentoml.Status$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.Status$Properties);

        /**
         * Status status_code.
         * @type {bentoml.Status.Code|undefined}
         */
        public status_code?: bentoml.Status.Code;

        /**
         * Status error_message.
         * @type {string|undefined}
         */
        public error_message?: string;

        /**
         * Creates a new Status instance using the specified properties.
         * @param {bentoml.Status$Properties=} [properties] Properties to set
         * @returns {bentoml.Status} Status instance
         */
        public static create(properties?: bentoml.Status$Properties): bentoml.Status;

        /**
         * Encodes the specified Status message. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param {bentoml.Status$Properties} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.Status$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Status message, length delimited. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param {bentoml.Status$Properties} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.Status$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Status message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Status} Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Status;

        /**
         * Decodes a Status message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Status} Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Status;

        /**
         * Verifies a Status message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Status} Status
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Status;

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Status.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Status} Status
         */
        public static from(object: { [k: string]: any }): bentoml.Status;

        /**
         * Creates a plain object from a Status message. Also converts values to other types if specified.
         * @param {bentoml.Status} message Status
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.Status, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this Status message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this Status to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace Status {

        /**
         * Code enum.
         * @name Code
         * @memberof bentoml.Status
         * @enum {number}
         * @property {number} OK=0 OK value
         * @property {number} CANCELLED=1 CANCELLED value
         * @property {number} UNKNOWN=2 UNKNOWN value
         * @property {number} INVALID_ARGUMENT=3 INVALID_ARGUMENT value
         * @property {number} DEADLINE_EXCEEDED=4 DEADLINE_EXCEEDED value
         * @property {number} NOT_FOUND=5 NOT_FOUND value
         * @property {number} ALREADY_EXISTS=6 ALREADY_EXISTS value
         * @property {number} PERMISSION_DENIED=7 PERMISSION_DENIED value
         * @property {number} UNAUTHENTICATED=16 UNAUTHENTICATED value
         * @property {number} RESOURCE_EXHAUSTED=8 RESOURCE_EXHAUSTED value
         * @property {number} FAILED_PRECONDITION=9 FAILED_PRECONDITION value
         * @property {number} ABORTED=10 ABORTED value
         * @property {number} OUT_OF_RANGE=11 OUT_OF_RANGE value
         * @property {number} UNIMPLEMENTED=12 UNIMPLEMENTED value
         * @property {number} INTERNAL=13 INTERNAL value
         * @property {number} UNAVAILABLE=14 UNAVAILABLE value
         * @property {number} DATA_LOSS=15 DATA_LOSS value
         * @property {number} DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_=20 DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ value
         */
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

    type BentoUri$Properties = {
        type?: bentoml.BentoUri.StorageType;
        uri?: string;
        cloud_presigned_url?: string;
    };

    /**
     * Constructs a new BentoUri.
     * @exports bentoml.BentoUri
     * @constructor
     * @param {bentoml.BentoUri$Properties=} [properties] Properties to set
     */
    class BentoUri {

        /**
         * Constructs a new BentoUri.
         * @exports bentoml.BentoUri
         * @constructor
         * @param {bentoml.BentoUri$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.BentoUri$Properties);

        /**
         * BentoUri type.
         * @type {bentoml.BentoUri.StorageType|undefined}
         */
        public type?: bentoml.BentoUri.StorageType;

        /**
         * BentoUri uri.
         * @type {string|undefined}
         */
        public uri?: string;

        /**
         * BentoUri cloud_presigned_url.
         * @type {string|undefined}
         */
        public cloud_presigned_url?: string;

        /**
         * Creates a new BentoUri instance using the specified properties.
         * @param {bentoml.BentoUri$Properties=} [properties] Properties to set
         * @returns {bentoml.BentoUri} BentoUri instance
         */
        public static create(properties?: bentoml.BentoUri$Properties): bentoml.BentoUri;

        /**
         * Encodes the specified BentoUri message. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param {bentoml.BentoUri$Properties} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.BentoUri$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified BentoUri message, length delimited. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param {bentoml.BentoUri$Properties} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.BentoUri$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a BentoUri message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.BentoUri} BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoUri;

        /**
         * Decodes a BentoUri message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoUri} BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoUri;

        /**
         * Verifies a BentoUri message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoUri} BentoUri
         */
        public static fromObject(object: { [k: string]: any }): bentoml.BentoUri;

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.BentoUri.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoUri} BentoUri
         */
        public static from(object: { [k: string]: any }): bentoml.BentoUri;

        /**
         * Creates a plain object from a BentoUri message. Also converts values to other types if specified.
         * @param {bentoml.BentoUri} message BentoUri
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.BentoUri, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this BentoUri message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this BentoUri to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace BentoUri {

        /**
         * StorageType enum.
         * @name StorageType
         * @memberof bentoml.BentoUri
         * @enum {number}
         * @property {number} UNSET=0 UNSET value
         * @property {number} LOCAL=1 LOCAL value
         * @property {number} S3=2 S3 value
         * @property {number} GCS=3 GCS value
         * @property {number} AZURE_BLOB_STORAGE=4 AZURE_BLOB_STORAGE value
         * @property {number} HDFS=5 HDFS value
         */
        enum StorageType {
            UNSET = 0,
            LOCAL = 1,
            S3 = 2,
            GCS = 3,
            AZURE_BLOB_STORAGE = 4,
            HDFS = 5
        }
    }

    type BentoServiceMetadata$Properties = {
        name?: string;
        version?: string;
        created_at?: google.protobuf.Timestamp$Properties;
        env?: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties;
        artifacts?: bentoml.BentoServiceMetadata.BentoArtifact$Properties[];
        apis?: bentoml.BentoServiceMetadata.BentoServiceApi$Properties[];
    };

    /**
     * Constructs a new BentoServiceMetadata.
     * @exports bentoml.BentoServiceMetadata
     * @constructor
     * @param {bentoml.BentoServiceMetadata$Properties=} [properties] Properties to set
     */
    class BentoServiceMetadata {

        /**
         * Constructs a new BentoServiceMetadata.
         * @exports bentoml.BentoServiceMetadata
         * @constructor
         * @param {bentoml.BentoServiceMetadata$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.BentoServiceMetadata$Properties);

        /**
         * BentoServiceMetadata name.
         * @type {string|undefined}
         */
        public name?: string;

        /**
         * BentoServiceMetadata version.
         * @type {string|undefined}
         */
        public version?: string;

        /**
         * BentoServiceMetadata created_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        public created_at?: google.protobuf.Timestamp$Properties;

        /**
         * BentoServiceMetadata env.
         * @type {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties|undefined}
         */
        public env?: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties;

        /**
         * BentoServiceMetadata artifacts.
         * @type {Array.<bentoml.BentoServiceMetadata.BentoArtifact$Properties>|undefined}
         */
        public artifacts?: bentoml.BentoServiceMetadata.BentoArtifact$Properties[];

        /**
         * BentoServiceMetadata apis.
         * @type {Array.<bentoml.BentoServiceMetadata.BentoServiceApi$Properties>|undefined}
         */
        public apis?: bentoml.BentoServiceMetadata.BentoServiceApi$Properties[];

        /**
         * Creates a new BentoServiceMetadata instance using the specified properties.
         * @param {bentoml.BentoServiceMetadata$Properties=} [properties] Properties to set
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata instance
         */
        public static create(properties?: bentoml.BentoServiceMetadata$Properties): bentoml.BentoServiceMetadata;

        /**
         * Encodes the specified BentoServiceMetadata message. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param {bentoml.BentoServiceMetadata$Properties} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.BentoServiceMetadata$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified BentoServiceMetadata message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param {bentoml.BentoServiceMetadata$Properties} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.BentoServiceMetadata$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata;

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata;

        /**
         * Verifies a BentoServiceMetadata message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a BentoServiceMetadata message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         */
        public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata;

        /**
         * Creates a BentoServiceMetadata message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.BentoServiceMetadata.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         */
        public static from(object: { [k: string]: any }): bentoml.BentoServiceMetadata;

        /**
         * Creates a plain object from a BentoServiceMetadata message. Also converts values to other types if specified.
         * @param {bentoml.BentoServiceMetadata} message BentoServiceMetadata
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.BentoServiceMetadata, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this BentoServiceMetadata message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this BentoServiceMetadata to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace BentoServiceMetadata {

        type BentoServiceEnv$Properties = {
            setup_sh?: string;
            conda_env?: string;
            pip_dependencies?: string;
            python_version?: string;
            docker_base_image?: string;
        };

        /**
         * Constructs a new BentoServiceEnv.
         * @exports bentoml.BentoServiceMetadata.BentoServiceEnv
         * @constructor
         * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties=} [properties] Properties to set
         */
        class BentoServiceEnv {

            /**
             * Constructs a new BentoServiceEnv.
             * @exports bentoml.BentoServiceMetadata.BentoServiceEnv
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties);

            /**
             * BentoServiceEnv setup_sh.
             * @type {string|undefined}
             */
            public setup_sh?: string;

            /**
             * BentoServiceEnv conda_env.
             * @type {string|undefined}
             */
            public conda_env?: string;

            /**
             * BentoServiceEnv pip_dependencies.
             * @type {string|undefined}
             */
            public pip_dependencies?: string;

            /**
             * BentoServiceEnv python_version.
             * @type {string|undefined}
             */
            public python_version?: string;

            /**
             * BentoServiceEnv docker_base_image.
             * @type {string|undefined}
             */
            public docker_base_image?: string;

            /**
             * Creates a new BentoServiceEnv instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Encodes the specified BentoServiceEnv message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoServiceEnv message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.BentoServiceEnv$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Verifies a BentoServiceEnv message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoServiceEnv.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             */
            public static from(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceEnv;

            /**
             * Creates a plain object from a BentoServiceEnv message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv} message BentoServiceEnv
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoServiceEnv, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this BentoServiceEnv message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoServiceEnv to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type BentoArtifact$Properties = {
            name?: string;
            artifact_type?: string;
        };

        /**
         * Constructs a new BentoArtifact.
         * @exports bentoml.BentoServiceMetadata.BentoArtifact
         * @constructor
         * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties=} [properties] Properties to set
         */
        class BentoArtifact {

            /**
             * Constructs a new BentoArtifact.
             * @exports bentoml.BentoServiceMetadata.BentoArtifact
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.BentoArtifact$Properties);

            /**
             * BentoArtifact name.
             * @type {string|undefined}
             */
            public name?: string;

            /**
             * BentoArtifact artifact_type.
             * @type {string|undefined}
             */
            public artifact_type?: string;

            /**
             * Creates a new BentoArtifact instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.BentoArtifact$Properties): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Encodes the specified BentoArtifact message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.BentoArtifact$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoArtifact message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.BentoArtifact$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Verifies a BentoArtifact message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoArtifact.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             */
            public static from(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoArtifact;

            /**
             * Creates a plain object from a BentoArtifact message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact} message BentoArtifact
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoArtifact, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this BentoArtifact message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoArtifact to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type BentoServiceApi$Properties = {
            name?: string;
            input_type?: string;
            docs?: string;
            input_config?: google.protobuf.Struct$Properties;
            output_config?: google.protobuf.Struct$Properties;
            output_type?: string;
            mb_max_latency?: number;
            mb_max_batch_size?: number;
        };

        /**
         * Constructs a new BentoServiceApi.
         * @exports bentoml.BentoServiceMetadata.BentoServiceApi
         * @constructor
         * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties=} [properties] Properties to set
         */
        class BentoServiceApi {

            /**
             * Constructs a new BentoServiceApi.
             * @exports bentoml.BentoServiceMetadata.BentoServiceApi
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties=} [properties] Properties to set
             */
            constructor(properties?: bentoml.BentoServiceMetadata.BentoServiceApi$Properties);

            /**
             * BentoServiceApi name.
             * @type {string|undefined}
             */
            public name?: string;

            /**
             * BentoServiceApi input_type.
             * @type {string|undefined}
             */
            public input_type?: string;

            /**
             * BentoServiceApi docs.
             * @type {string|undefined}
             */
            public docs?: string;

            /**
             * BentoServiceApi input_config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            public input_config?: google.protobuf.Struct$Properties;

            /**
             * BentoServiceApi output_config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            public output_config?: google.protobuf.Struct$Properties;

            /**
             * BentoServiceApi output_type.
             * @type {string|undefined}
             */
            public output_type?: string;

            /**
             * BentoServiceApi mb_max_latency.
             * @type {number|undefined}
             */
            public mb_max_latency?: number;

            /**
             * BentoServiceApi mb_max_batch_size.
             * @type {number|undefined}
             */
            public mb_max_batch_size?: number;

            /**
             * Creates a new BentoServiceApi instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi instance
             */
            public static create(properties?: bentoml.BentoServiceMetadata.BentoServiceApi$Properties): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Encodes the specified BentoServiceApi message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: bentoml.BentoServiceMetadata.BentoServiceApi$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BentoServiceApi message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: bentoml.BentoServiceMetadata.BentoServiceApi$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Verifies a BentoServiceApi message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             */
            public static fromObject(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoServiceApi.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             */
            public static from(object: { [k: string]: any }): bentoml.BentoServiceMetadata.BentoServiceApi;

            /**
             * Creates a plain object from a BentoServiceApi message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi} message BentoServiceApi
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: bentoml.BentoServiceMetadata.BentoServiceApi, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this BentoServiceApi message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this BentoServiceApi to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    type Bento$Properties = {
        name?: string;
        version?: string;
        uri?: bentoml.BentoUri$Properties;
        bento_service_metadata?: bentoml.BentoServiceMetadata$Properties;
        status?: bentoml.UploadStatus$Properties;
    };

    /**
     * Constructs a new Bento.
     * @exports bentoml.Bento
     * @constructor
     * @param {bentoml.Bento$Properties=} [properties] Properties to set
     */
    class Bento {

        /**
         * Constructs a new Bento.
         * @exports bentoml.Bento
         * @constructor
         * @param {bentoml.Bento$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.Bento$Properties);

        /**
         * Bento name.
         * @type {string|undefined}
         */
        public name?: string;

        /**
         * Bento version.
         * @type {string|undefined}
         */
        public version?: string;

        /**
         * Bento uri.
         * @type {bentoml.BentoUri$Properties|undefined}
         */
        public uri?: bentoml.BentoUri$Properties;

        /**
         * Bento bento_service_metadata.
         * @type {bentoml.BentoServiceMetadata$Properties|undefined}
         */
        public bento_service_metadata?: bentoml.BentoServiceMetadata$Properties;

        /**
         * Bento status.
         * @type {bentoml.UploadStatus$Properties|undefined}
         */
        public status?: bentoml.UploadStatus$Properties;

        /**
         * Creates a new Bento instance using the specified properties.
         * @param {bentoml.Bento$Properties=} [properties] Properties to set
         * @returns {bentoml.Bento} Bento instance
         */
        public static create(properties?: bentoml.Bento$Properties): bentoml.Bento;

        /**
         * Encodes the specified Bento message. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param {bentoml.Bento$Properties} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.Bento$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Bento message, length delimited. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param {bentoml.Bento$Properties} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.Bento$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Bento message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Bento} Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Bento;

        /**
         * Decodes a Bento message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Bento} Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Bento;

        /**
         * Verifies a Bento message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Bento} Bento
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Bento;

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Bento.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Bento} Bento
         */
        public static from(object: { [k: string]: any }): bentoml.Bento;

        /**
         * Creates a plain object from a Bento message. Also converts values to other types if specified.
         * @param {bentoml.Bento} message Bento
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.Bento, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this Bento message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this Bento to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type AddBentoRequest$Properties = {
        bento_name?: string;
        bento_version?: string;
    };

    /**
     * Constructs a new AddBentoRequest.
     * @exports bentoml.AddBentoRequest
     * @constructor
     * @param {bentoml.AddBentoRequest$Properties=} [properties] Properties to set
     */
    class AddBentoRequest {

        /**
         * Constructs a new AddBentoRequest.
         * @exports bentoml.AddBentoRequest
         * @constructor
         * @param {bentoml.AddBentoRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.AddBentoRequest$Properties);

        /**
         * AddBentoRequest bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * AddBentoRequest bento_version.
         * @type {string|undefined}
         */
        public bento_version?: string;

        /**
         * Creates a new AddBentoRequest instance using the specified properties.
         * @param {bentoml.AddBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.AddBentoRequest} AddBentoRequest instance
         */
        public static create(properties?: bentoml.AddBentoRequest$Properties): bentoml.AddBentoRequest;

        /**
         * Encodes the specified AddBentoRequest message. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param {bentoml.AddBentoRequest$Properties} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.AddBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AddBentoRequest message, length delimited. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param {bentoml.AddBentoRequest$Properties} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.AddBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.AddBentoRequest;

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.AddBentoRequest;

        /**
         * Verifies an AddBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.AddBentoRequest;

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.AddBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         */
        public static from(object: { [k: string]: any }): bentoml.AddBentoRequest;

        /**
         * Creates a plain object from an AddBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.AddBentoRequest} message AddBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.AddBentoRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this AddBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this AddBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type AddBentoResponse$Properties = {
        status?: bentoml.Status$Properties;
        uri?: bentoml.BentoUri$Properties;
    };

    /**
     * Constructs a new AddBentoResponse.
     * @exports bentoml.AddBentoResponse
     * @constructor
     * @param {bentoml.AddBentoResponse$Properties=} [properties] Properties to set
     */
    class AddBentoResponse {

        /**
         * Constructs a new AddBentoResponse.
         * @exports bentoml.AddBentoResponse
         * @constructor
         * @param {bentoml.AddBentoResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.AddBentoResponse$Properties);

        /**
         * AddBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * AddBentoResponse uri.
         * @type {bentoml.BentoUri$Properties|undefined}
         */
        public uri?: bentoml.BentoUri$Properties;

        /**
         * Creates a new AddBentoResponse instance using the specified properties.
         * @param {bentoml.AddBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.AddBentoResponse} AddBentoResponse instance
         */
        public static create(properties?: bentoml.AddBentoResponse$Properties): bentoml.AddBentoResponse;

        /**
         * Encodes the specified AddBentoResponse message. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param {bentoml.AddBentoResponse$Properties} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.AddBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AddBentoResponse message, length delimited. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param {bentoml.AddBentoResponse$Properties} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.AddBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.AddBentoResponse;

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.AddBentoResponse;

        /**
         * Verifies an AddBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.AddBentoResponse;

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.AddBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         */
        public static from(object: { [k: string]: any }): bentoml.AddBentoResponse;

        /**
         * Creates a plain object from an AddBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.AddBentoResponse} message AddBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.AddBentoResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this AddBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this AddBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type UploadStatus$Properties = {
        status?: bentoml.UploadStatus.Status;
        updated_at?: google.protobuf.Timestamp$Properties;
        percentage?: number;
        error_message?: string;
    };

    /**
     * Constructs a new UploadStatus.
     * @exports bentoml.UploadStatus
     * @constructor
     * @param {bentoml.UploadStatus$Properties=} [properties] Properties to set
     */
    class UploadStatus {

        /**
         * Constructs a new UploadStatus.
         * @exports bentoml.UploadStatus
         * @constructor
         * @param {bentoml.UploadStatus$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.UploadStatus$Properties);

        /**
         * UploadStatus status.
         * @type {bentoml.UploadStatus.Status|undefined}
         */
        public status?: bentoml.UploadStatus.Status;

        /**
         * UploadStatus updated_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        public updated_at?: google.protobuf.Timestamp$Properties;

        /**
         * UploadStatus percentage.
         * @type {number|undefined}
         */
        public percentage?: number;

        /**
         * UploadStatus error_message.
         * @type {string|undefined}
         */
        public error_message?: string;

        /**
         * Creates a new UploadStatus instance using the specified properties.
         * @param {bentoml.UploadStatus$Properties=} [properties] Properties to set
         * @returns {bentoml.UploadStatus} UploadStatus instance
         */
        public static create(properties?: bentoml.UploadStatus$Properties): bentoml.UploadStatus;

        /**
         * Encodes the specified UploadStatus message. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param {bentoml.UploadStatus$Properties} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.UploadStatus$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UploadStatus message, length delimited. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param {bentoml.UploadStatus$Properties} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.UploadStatus$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UploadStatus message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UploadStatus} UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UploadStatus;

        /**
         * Decodes an UploadStatus message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UploadStatus} UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UploadStatus;

        /**
         * Verifies an UploadStatus message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UploadStatus} UploadStatus
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UploadStatus;

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UploadStatus.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UploadStatus} UploadStatus
         */
        public static from(object: { [k: string]: any }): bentoml.UploadStatus;

        /**
         * Creates a plain object from an UploadStatus message. Also converts values to other types if specified.
         * @param {bentoml.UploadStatus} message UploadStatus
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.UploadStatus, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this UploadStatus message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this UploadStatus to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace UploadStatus {

        /**
         * Status enum.
         * @name Status
         * @memberof bentoml.UploadStatus
         * @enum {number}
         * @property {number} UNINITIALIZED=0 UNINITIALIZED value
         * @property {number} UPLOADING=1 UPLOADING value
         * @property {number} DONE=2 DONE value
         * @property {number} ERROR=3 ERROR value
         * @property {number} TIMEOUT=4 TIMEOUT value
         */
        enum Status {
            UNINITIALIZED = 0,
            UPLOADING = 1,
            DONE = 2,
            ERROR = 3,
            TIMEOUT = 4
        }
    }

    type UpdateBentoRequest$Properties = {
        bento_name?: string;
        bento_version?: string;
        upload_status?: bentoml.UploadStatus$Properties;
        service_metadata?: bentoml.BentoServiceMetadata$Properties;
    };

    /**
     * Constructs a new UpdateBentoRequest.
     * @exports bentoml.UpdateBentoRequest
     * @constructor
     * @param {bentoml.UpdateBentoRequest$Properties=} [properties] Properties to set
     */
    class UpdateBentoRequest {

        /**
         * Constructs a new UpdateBentoRequest.
         * @exports bentoml.UpdateBentoRequest
         * @constructor
         * @param {bentoml.UpdateBentoRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.UpdateBentoRequest$Properties);

        /**
         * UpdateBentoRequest bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * UpdateBentoRequest bento_version.
         * @type {string|undefined}
         */
        public bento_version?: string;

        /**
         * UpdateBentoRequest upload_status.
         * @type {bentoml.UploadStatus$Properties|undefined}
         */
        public upload_status?: bentoml.UploadStatus$Properties;

        /**
         * UpdateBentoRequest service_metadata.
         * @type {bentoml.BentoServiceMetadata$Properties|undefined}
         */
        public service_metadata?: bentoml.BentoServiceMetadata$Properties;

        /**
         * Creates a new UpdateBentoRequest instance using the specified properties.
         * @param {bentoml.UpdateBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest instance
         */
        public static create(properties?: bentoml.UpdateBentoRequest$Properties): bentoml.UpdateBentoRequest;

        /**
         * Encodes the specified UpdateBentoRequest message. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param {bentoml.UpdateBentoRequest$Properties} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.UpdateBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UpdateBentoRequest message, length delimited. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param {bentoml.UpdateBentoRequest$Properties} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.UpdateBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UpdateBentoRequest;

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UpdateBentoRequest;

        /**
         * Verifies an UpdateBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UpdateBentoRequest;

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UpdateBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         */
        public static from(object: { [k: string]: any }): bentoml.UpdateBentoRequest;

        /**
         * Creates a plain object from an UpdateBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.UpdateBentoRequest} message UpdateBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.UpdateBentoRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this UpdateBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this UpdateBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type UpdateBentoResponse$Properties = {
        status?: bentoml.Status$Properties;
    };

    /**
     * Constructs a new UpdateBentoResponse.
     * @exports bentoml.UpdateBentoResponse
     * @constructor
     * @param {bentoml.UpdateBentoResponse$Properties=} [properties] Properties to set
     */
    class UpdateBentoResponse {

        /**
         * Constructs a new UpdateBentoResponse.
         * @exports bentoml.UpdateBentoResponse
         * @constructor
         * @param {bentoml.UpdateBentoResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.UpdateBentoResponse$Properties);

        /**
         * UpdateBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * Creates a new UpdateBentoResponse instance using the specified properties.
         * @param {bentoml.UpdateBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse instance
         */
        public static create(properties?: bentoml.UpdateBentoResponse$Properties): bentoml.UpdateBentoResponse;

        /**
         * Encodes the specified UpdateBentoResponse message. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param {bentoml.UpdateBentoResponse$Properties} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.UpdateBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified UpdateBentoResponse message, length delimited. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param {bentoml.UpdateBentoResponse$Properties} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.UpdateBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.UpdateBentoResponse;

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.UpdateBentoResponse;

        /**
         * Verifies an UpdateBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.UpdateBentoResponse;

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UpdateBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         */
        public static from(object: { [k: string]: any }): bentoml.UpdateBentoResponse;

        /**
         * Creates a plain object from an UpdateBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.UpdateBentoResponse} message UpdateBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.UpdateBentoResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this UpdateBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this UpdateBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DangerouslyDeleteBentoRequest$Properties = {
        bento_name?: string;
        bento_version?: string;
    };

    /**
     * Constructs a new DangerouslyDeleteBentoRequest.
     * @exports bentoml.DangerouslyDeleteBentoRequest
     * @constructor
     * @param {bentoml.DangerouslyDeleteBentoRequest$Properties=} [properties] Properties to set
     */
    class DangerouslyDeleteBentoRequest {

        /**
         * Constructs a new DangerouslyDeleteBentoRequest.
         * @exports bentoml.DangerouslyDeleteBentoRequest
         * @constructor
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DangerouslyDeleteBentoRequest$Properties);

        /**
         * DangerouslyDeleteBentoRequest bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * DangerouslyDeleteBentoRequest bento_version.
         * @type {string|undefined}
         */
        public bento_version?: string;

        /**
         * Creates a new DangerouslyDeleteBentoRequest instance using the specified properties.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest instance
         */
        public static create(properties?: bentoml.DangerouslyDeleteBentoRequest$Properties): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DangerouslyDeleteBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DangerouslyDeleteBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Verifies a DangerouslyDeleteBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DangerouslyDeleteBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         */
        public static from(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoRequest;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.DangerouslyDeleteBentoRequest} message DangerouslyDeleteBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DangerouslyDeleteBentoRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DangerouslyDeleteBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type DangerouslyDeleteBentoResponse$Properties = {
        status?: bentoml.Status$Properties;
    };

    /**
     * Constructs a new DangerouslyDeleteBentoResponse.
     * @exports bentoml.DangerouslyDeleteBentoResponse
     * @constructor
     * @param {bentoml.DangerouslyDeleteBentoResponse$Properties=} [properties] Properties to set
     */
    class DangerouslyDeleteBentoResponse {

        /**
         * Constructs a new DangerouslyDeleteBentoResponse.
         * @exports bentoml.DangerouslyDeleteBentoResponse
         * @constructor
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.DangerouslyDeleteBentoResponse$Properties);

        /**
         * DangerouslyDeleteBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * Creates a new DangerouslyDeleteBentoResponse instance using the specified properties.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse instance
         */
        public static create(properties?: bentoml.DangerouslyDeleteBentoResponse$Properties): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.DangerouslyDeleteBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.DangerouslyDeleteBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Verifies a DangerouslyDeleteBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DangerouslyDeleteBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         */
        public static from(object: { [k: string]: any }): bentoml.DangerouslyDeleteBentoResponse;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.DangerouslyDeleteBentoResponse} message DangerouslyDeleteBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.DangerouslyDeleteBentoResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this DangerouslyDeleteBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type GetBentoRequest$Properties = {
        bento_name?: string;
        bento_version?: string;
    };

    /**
     * Constructs a new GetBentoRequest.
     * @exports bentoml.GetBentoRequest
     * @constructor
     * @param {bentoml.GetBentoRequest$Properties=} [properties] Properties to set
     */
    class GetBentoRequest {

        /**
         * Constructs a new GetBentoRequest.
         * @exports bentoml.GetBentoRequest
         * @constructor
         * @param {bentoml.GetBentoRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.GetBentoRequest$Properties);

        /**
         * GetBentoRequest bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * GetBentoRequest bento_version.
         * @type {string|undefined}
         */
        public bento_version?: string;

        /**
         * Creates a new GetBentoRequest instance using the specified properties.
         * @param {bentoml.GetBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.GetBentoRequest} GetBentoRequest instance
         */
        public static create(properties?: bentoml.GetBentoRequest$Properties): bentoml.GetBentoRequest;

        /**
         * Encodes the specified GetBentoRequest message. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param {bentoml.GetBentoRequest$Properties} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.GetBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetBentoRequest message, length delimited. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param {bentoml.GetBentoRequest$Properties} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.GetBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetBentoRequest;

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetBentoRequest;

        /**
         * Verifies a GetBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetBentoRequest;

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         */
        public static from(object: { [k: string]: any }): bentoml.GetBentoRequest;

        /**
         * Creates a plain object from a GetBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.GetBentoRequest} message GetBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.GetBentoRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this GetBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this GetBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type GetBentoResponse$Properties = {
        status?: bentoml.Status$Properties;
        bento?: bentoml.Bento$Properties;
    };

    /**
     * Constructs a new GetBentoResponse.
     * @exports bentoml.GetBentoResponse
     * @constructor
     * @param {bentoml.GetBentoResponse$Properties=} [properties] Properties to set
     */
    class GetBentoResponse {

        /**
         * Constructs a new GetBentoResponse.
         * @exports bentoml.GetBentoResponse
         * @constructor
         * @param {bentoml.GetBentoResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.GetBentoResponse$Properties);

        /**
         * GetBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * GetBentoResponse bento.
         * @type {bentoml.Bento$Properties|undefined}
         */
        public bento?: bentoml.Bento$Properties;

        /**
         * Creates a new GetBentoResponse instance using the specified properties.
         * @param {bentoml.GetBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetBentoResponse} GetBentoResponse instance
         */
        public static create(properties?: bentoml.GetBentoResponse$Properties): bentoml.GetBentoResponse;

        /**
         * Encodes the specified GetBentoResponse message. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param {bentoml.GetBentoResponse$Properties} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.GetBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetBentoResponse message, length delimited. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param {bentoml.GetBentoResponse$Properties} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.GetBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetBentoResponse;

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetBentoResponse;

        /**
         * Verifies a GetBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetBentoResponse;

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         */
        public static from(object: { [k: string]: any }): bentoml.GetBentoResponse;

        /**
         * Creates a plain object from a GetBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetBentoResponse} message GetBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.GetBentoResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this GetBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this GetBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type ListBentoRequest$Properties = {
        bento_name?: string;
        offset?: number;
        limit?: number;
        order_by?: bentoml.ListBentoRequest.SORTABLE_COLUMN;
        ascending_order?: boolean;
    };

    /**
     * Constructs a new ListBentoRequest.
     * @exports bentoml.ListBentoRequest
     * @constructor
     * @param {bentoml.ListBentoRequest$Properties=} [properties] Properties to set
     */
    class ListBentoRequest {

        /**
         * Constructs a new ListBentoRequest.
         * @exports bentoml.ListBentoRequest
         * @constructor
         * @param {bentoml.ListBentoRequest$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ListBentoRequest$Properties);

        /**
         * ListBentoRequest bento_name.
         * @type {string|undefined}
         */
        public bento_name?: string;

        /**
         * ListBentoRequest offset.
         * @type {number|undefined}
         */
        public offset?: number;

        /**
         * ListBentoRequest limit.
         * @type {number|undefined}
         */
        public limit?: number;

        /**
         * ListBentoRequest order_by.
         * @type {bentoml.ListBentoRequest.SORTABLE_COLUMN|undefined}
         */
        public order_by?: bentoml.ListBentoRequest.SORTABLE_COLUMN;

        /**
         * ListBentoRequest ascending_order.
         * @type {boolean|undefined}
         */
        public ascending_order?: boolean;

        /**
         * Creates a new ListBentoRequest instance using the specified properties.
         * @param {bentoml.ListBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ListBentoRequest} ListBentoRequest instance
         */
        public static create(properties?: bentoml.ListBentoRequest$Properties): bentoml.ListBentoRequest;

        /**
         * Encodes the specified ListBentoRequest message. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param {bentoml.ListBentoRequest$Properties} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ListBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListBentoRequest message, length delimited. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param {bentoml.ListBentoRequest$Properties} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ListBentoRequest$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListBentoRequest;

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListBentoRequest;

        /**
         * Verifies a ListBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListBentoRequest;

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         */
        public static from(object: { [k: string]: any }): bentoml.ListBentoRequest;

        /**
         * Creates a plain object from a ListBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.ListBentoRequest} message ListBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ListBentoRequest, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ListBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ListBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace ListBentoRequest {

        /**
         * SORTABLE_COLUMN enum.
         * @name SORTABLE_COLUMN
         * @memberof bentoml.ListBentoRequest
         * @enum {number}
         * @property {number} created_at=0 created_at value
         * @property {number} name=1 name value
         */
        enum SORTABLE_COLUMN {
            created_at = 0,
            name = 1
        }
    }

    type ListBentoResponse$Properties = {
        status?: bentoml.Status$Properties;
        bentos?: bentoml.Bento$Properties[];
    };

    /**
     * Constructs a new ListBentoResponse.
     * @exports bentoml.ListBentoResponse
     * @constructor
     * @param {bentoml.ListBentoResponse$Properties=} [properties] Properties to set
     */
    class ListBentoResponse {

        /**
         * Constructs a new ListBentoResponse.
         * @exports bentoml.ListBentoResponse
         * @constructor
         * @param {bentoml.ListBentoResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.ListBentoResponse$Properties);

        /**
         * ListBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * ListBentoResponse bentos.
         * @type {Array.<bentoml.Bento$Properties>|undefined}
         */
        public bentos?: bentoml.Bento$Properties[];

        /**
         * Creates a new ListBentoResponse instance using the specified properties.
         * @param {bentoml.ListBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ListBentoResponse} ListBentoResponse instance
         */
        public static create(properties?: bentoml.ListBentoResponse$Properties): bentoml.ListBentoResponse;

        /**
         * Encodes the specified ListBentoResponse message. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param {bentoml.ListBentoResponse$Properties} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.ListBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified ListBentoResponse message, length delimited. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param {bentoml.ListBentoResponse$Properties} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.ListBentoResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.ListBentoResponse;

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.ListBentoResponse;

        /**
         * Verifies a ListBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a ListBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.ListBentoResponse;

        /**
         * Creates a ListBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         */
        public static from(object: { [k: string]: any }): bentoml.ListBentoResponse;

        /**
         * Creates a plain object from a ListBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.ListBentoResponse} message ListBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.ListBentoResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this ListBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this ListBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /**
     * Constructs a new Yatai service.
     * @exports bentoml.Yatai
     * @extends $protobuf.rpc.Service
     * @constructor
     * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
     * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
     * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
     */
    class Yatai extends $protobuf.rpc.Service {

        /**
         * Constructs a new Yatai service.
         * @exports bentoml.Yatai
         * @extends $protobuf.rpc.Service
         * @constructor
         * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
         * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
         * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
         */
        constructor(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean);

        /**
         * Creates new Yatai service using the specified rpc implementation.
         * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
         * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
         * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
         * @returns {Yatai} RPC service. Useful where requests and/or responses are streamed.
         */
        public static create(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean): Yatai;

        /**
         * Calls HealthCheck.
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @param {Yatai_healthCheck_Callback} callback Node-style callback called with the error, if any, and HealthCheckResponse
         * @returns {undefined}
         */
        public healthCheck(request: (google.protobuf.Empty|{ [k: string]: any }), callback: Yatai_healthCheck_Callback): void;

        /**
         * Calls GetYataiServiceVersion.
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @param {Yatai_getYataiServiceVersion_Callback} callback Node-style callback called with the error, if any, and GetYataiServiceVersionResponse
         * @returns {undefined}
         */
        public getYataiServiceVersion(request: (google.protobuf.Empty|{ [k: string]: any }), callback: Yatai_getYataiServiceVersion_Callback): void;

        /**
         * Calls ApplyDeployment.
         * @param {bentoml.ApplyDeploymentRequest|Object.<string,*>} request ApplyDeploymentRequest message or plain object
         * @param {Yatai_applyDeployment_Callback} callback Node-style callback called with the error, if any, and ApplyDeploymentResponse
         * @returns {undefined}
         */
        public applyDeployment(request: (bentoml.ApplyDeploymentRequest|{ [k: string]: any }), callback: Yatai_applyDeployment_Callback): void;

        /**
         * Calls DeleteDeployment.
         * @param {bentoml.DeleteDeploymentRequest|Object.<string,*>} request DeleteDeploymentRequest message or plain object
         * @param {Yatai_deleteDeployment_Callback} callback Node-style callback called with the error, if any, and DeleteDeploymentResponse
         * @returns {undefined}
         */
        public deleteDeployment(request: (bentoml.DeleteDeploymentRequest|{ [k: string]: any }), callback: Yatai_deleteDeployment_Callback): void;

        /**
         * Calls GetDeployment.
         * @param {bentoml.GetDeploymentRequest|Object.<string,*>} request GetDeploymentRequest message or plain object
         * @param {Yatai_getDeployment_Callback} callback Node-style callback called with the error, if any, and GetDeploymentResponse
         * @returns {undefined}
         */
        public getDeployment(request: (bentoml.GetDeploymentRequest|{ [k: string]: any }), callback: Yatai_getDeployment_Callback): void;

        /**
         * Calls DescribeDeployment.
         * @param {bentoml.DescribeDeploymentRequest|Object.<string,*>} request DescribeDeploymentRequest message or plain object
         * @param {Yatai_describeDeployment_Callback} callback Node-style callback called with the error, if any, and DescribeDeploymentResponse
         * @returns {undefined}
         */
        public describeDeployment(request: (bentoml.DescribeDeploymentRequest|{ [k: string]: any }), callback: Yatai_describeDeployment_Callback): void;

        /**
         * Calls ListDeployments.
         * @param {bentoml.ListDeploymentsRequest|Object.<string,*>} request ListDeploymentsRequest message or plain object
         * @param {Yatai_listDeployments_Callback} callback Node-style callback called with the error, if any, and ListDeploymentsResponse
         * @returns {undefined}
         */
        public listDeployments(request: (bentoml.ListDeploymentsRequest|{ [k: string]: any }), callback: Yatai_listDeployments_Callback): void;

        /**
         * Calls AddBento.
         * @param {bentoml.AddBentoRequest|Object.<string,*>} request AddBentoRequest message or plain object
         * @param {Yatai_addBento_Callback} callback Node-style callback called with the error, if any, and AddBentoResponse
         * @returns {undefined}
         */
        public addBento(request: (bentoml.AddBentoRequest|{ [k: string]: any }), callback: Yatai_addBento_Callback): void;

        /**
         * Calls UpdateBento.
         * @param {bentoml.UpdateBentoRequest|Object.<string,*>} request UpdateBentoRequest message or plain object
         * @param {Yatai_updateBento_Callback} callback Node-style callback called with the error, if any, and UpdateBentoResponse
         * @returns {undefined}
         */
        public updateBento(request: (bentoml.UpdateBentoRequest|{ [k: string]: any }), callback: Yatai_updateBento_Callback): void;

        /**
         * Calls GetBento.
         * @param {bentoml.GetBentoRequest|Object.<string,*>} request GetBentoRequest message or plain object
         * @param {Yatai_getBento_Callback} callback Node-style callback called with the error, if any, and GetBentoResponse
         * @returns {undefined}
         */
        public getBento(request: (bentoml.GetBentoRequest|{ [k: string]: any }), callback: Yatai_getBento_Callback): void;

        /**
         * Calls DangerouslyDeleteBento.
         * @param {bentoml.DangerouslyDeleteBentoRequest|Object.<string,*>} request DangerouslyDeleteBentoRequest message or plain object
         * @param {Yatai_dangerouslyDeleteBento_Callback} callback Node-style callback called with the error, if any, and DangerouslyDeleteBentoResponse
         * @returns {undefined}
         */
        public dangerouslyDeleteBento(request: (bentoml.DangerouslyDeleteBentoRequest|{ [k: string]: any }), callback: Yatai_dangerouslyDeleteBento_Callback): void;

        /**
         * Calls ListBento.
         * @param {bentoml.ListBentoRequest|Object.<string,*>} request ListBentoRequest message or plain object
         * @param {Yatai_listBento_Callback} callback Node-style callback called with the error, if any, and ListBentoResponse
         * @returns {undefined}
         */
        public listBento(request: (bentoml.ListBentoRequest|{ [k: string]: any }), callback: Yatai_listBento_Callback): void;
    }

    type HealthCheckResponse$Properties = {
        status?: bentoml.Status$Properties;
    };

    /**
     * Constructs a new HealthCheckResponse.
     * @exports bentoml.HealthCheckResponse
     * @constructor
     * @param {bentoml.HealthCheckResponse$Properties=} [properties] Properties to set
     */
    class HealthCheckResponse {

        /**
         * Constructs a new HealthCheckResponse.
         * @exports bentoml.HealthCheckResponse
         * @constructor
         * @param {bentoml.HealthCheckResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.HealthCheckResponse$Properties);

        /**
         * HealthCheckResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * Creates a new HealthCheckResponse instance using the specified properties.
         * @param {bentoml.HealthCheckResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse instance
         */
        public static create(properties?: bentoml.HealthCheckResponse$Properties): bentoml.HealthCheckResponse;

        /**
         * Encodes the specified HealthCheckResponse message. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param {bentoml.HealthCheckResponse$Properties} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.HealthCheckResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified HealthCheckResponse message, length delimited. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param {bentoml.HealthCheckResponse$Properties} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.HealthCheckResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.HealthCheckResponse;

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.HealthCheckResponse;

        /**
         * Verifies a HealthCheckResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.HealthCheckResponse;

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.HealthCheckResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         */
        public static from(object: { [k: string]: any }): bentoml.HealthCheckResponse;

        /**
         * Creates a plain object from a HealthCheckResponse message. Also converts values to other types if specified.
         * @param {bentoml.HealthCheckResponse} message HealthCheckResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.HealthCheckResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this HealthCheckResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this HealthCheckResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type GetYataiServiceVersionResponse$Properties = {
        status?: bentoml.Status$Properties;
        version?: string;
    };

    /**
     * Constructs a new GetYataiServiceVersionResponse.
     * @exports bentoml.GetYataiServiceVersionResponse
     * @constructor
     * @param {bentoml.GetYataiServiceVersionResponse$Properties=} [properties] Properties to set
     */
    class GetYataiServiceVersionResponse {

        /**
         * Constructs a new GetYataiServiceVersionResponse.
         * @exports bentoml.GetYataiServiceVersionResponse
         * @constructor
         * @param {bentoml.GetYataiServiceVersionResponse$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.GetYataiServiceVersionResponse$Properties);

        /**
         * GetYataiServiceVersionResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        public status?: bentoml.Status$Properties;

        /**
         * GetYataiServiceVersionResponse version.
         * @type {string|undefined}
         */
        public version?: string;

        /**
         * Creates a new GetYataiServiceVersionResponse instance using the specified properties.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse instance
         */
        public static create(properties?: bentoml.GetYataiServiceVersionResponse$Properties): bentoml.GetYataiServiceVersionResponse;

        /**
         * Encodes the specified GetYataiServiceVersionResponse message. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.GetYataiServiceVersionResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GetYataiServiceVersionResponse message, length delimited. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.GetYataiServiceVersionResponse$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.GetYataiServiceVersionResponse;

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.GetYataiServiceVersionResponse;

        /**
         * Verifies a GetYataiServiceVersionResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         */
        public static fromObject(object: { [k: string]: any }): bentoml.GetYataiServiceVersionResponse;

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetYataiServiceVersionResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         */
        public static from(object: { [k: string]: any }): bentoml.GetYataiServiceVersionResponse;

        /**
         * Creates a plain object from a GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetYataiServiceVersionResponse} message GetYataiServiceVersionResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.GetYataiServiceVersionResponse, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this GetYataiServiceVersionResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    type Chunk$Properties = {
        content?: Uint8Array;
    };

    /**
     * Constructs a new Chunk.
     * @exports bentoml.Chunk
     * @constructor
     * @param {bentoml.Chunk$Properties=} [properties] Properties to set
     */
    class Chunk {

        /**
         * Constructs a new Chunk.
         * @exports bentoml.Chunk
         * @constructor
         * @param {bentoml.Chunk$Properties=} [properties] Properties to set
         */
        constructor(properties?: bentoml.Chunk$Properties);

        /**
         * Chunk content.
         * @type {Uint8Array|undefined}
         */
        public content?: Uint8Array;

        /**
         * Creates a new Chunk instance using the specified properties.
         * @param {bentoml.Chunk$Properties=} [properties] Properties to set
         * @returns {bentoml.Chunk} Chunk instance
         */
        public static create(properties?: bentoml.Chunk$Properties): bentoml.Chunk;

        /**
         * Encodes the specified Chunk message. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param {bentoml.Chunk$Properties} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encode(message: bentoml.Chunk$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Chunk message, length delimited. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param {bentoml.Chunk$Properties} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        public static encodeDelimited(message: bentoml.Chunk$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Chunk message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Chunk} Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): bentoml.Chunk;

        /**
         * Decodes a Chunk message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Chunk} Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): bentoml.Chunk;

        /**
         * Verifies a Chunk message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): string;

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Chunk} Chunk
         */
        public static fromObject(object: { [k: string]: any }): bentoml.Chunk;

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Chunk.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Chunk} Chunk
         */
        public static from(object: { [k: string]: any }): bentoml.Chunk;

        /**
         * Creates a plain object from a Chunk message. Also converts values to other types if specified.
         * @param {bentoml.Chunk} message Chunk
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public static toObject(message: bentoml.Chunk, options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Creates a plain object from this Chunk message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

        /**
         * Converts this Chunk to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        public toJSON(): { [k: string]: any };
    }
}

type Yatai_healthCheck_Callback = (error: Error, response?: bentoml.HealthCheckResponse) => void;

type Yatai_getYataiServiceVersion_Callback = (error: Error, response?: bentoml.GetYataiServiceVersionResponse) => void;

type Yatai_applyDeployment_Callback = (error: Error, response?: bentoml.ApplyDeploymentResponse) => void;

type Yatai_deleteDeployment_Callback = (error: Error, response?: bentoml.DeleteDeploymentResponse) => void;

type Yatai_getDeployment_Callback = (error: Error, response?: bentoml.GetDeploymentResponse) => void;

type Yatai_describeDeployment_Callback = (error: Error, response?: bentoml.DescribeDeploymentResponse) => void;

type Yatai_listDeployments_Callback = (error: Error, response?: bentoml.ListDeploymentsResponse) => void;

type Yatai_addBento_Callback = (error: Error, response?: bentoml.AddBentoResponse) => void;

type Yatai_updateBento_Callback = (error: Error, response?: bentoml.UpdateBentoResponse) => void;

type Yatai_getBento_Callback = (error: Error, response?: bentoml.GetBentoResponse) => void;

type Yatai_dangerouslyDeleteBento_Callback = (error: Error, response?: bentoml.DangerouslyDeleteBentoResponse) => void;

type Yatai_listBento_Callback = (error: Error, response?: bentoml.ListBentoResponse) => void;

/**
 * Namespace google.
 * @exports google
 * @namespace
 */
export namespace google {

    /**
     * Namespace protobuf.
     * @exports google.protobuf
     * @namespace
     */
    namespace protobuf {

        type Struct$Properties = {
            fields?: { [k: string]: google.protobuf.Value$Properties };
        };

        /**
         * Constructs a new Struct.
         * @exports google.protobuf.Struct
         * @constructor
         * @param {google.protobuf.Struct$Properties=} [properties] Properties to set
         */
        class Struct {

            /**
             * Constructs a new Struct.
             * @exports google.protobuf.Struct
             * @constructor
             * @param {google.protobuf.Struct$Properties=} [properties] Properties to set
             */
            constructor(properties?: google.protobuf.Struct$Properties);

            /**
             * Struct fields.
             * @type {Object.<string,google.protobuf.Value$Properties>|undefined}
             */
            public fields?: { [k: string]: google.protobuf.Value$Properties };

            /**
             * Creates a new Struct instance using the specified properties.
             * @param {google.protobuf.Struct$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Struct} Struct instance
             */
            public static create(properties?: google.protobuf.Struct$Properties): google.protobuf.Struct;

            /**
             * Encodes the specified Struct message. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param {google.protobuf.Struct$Properties} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: google.protobuf.Struct$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Struct message, length delimited. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param {google.protobuf.Struct$Properties} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: google.protobuf.Struct$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Struct message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Struct} Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Struct;

            /**
             * Decodes a Struct message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Struct} Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Struct;

            /**
             * Verifies a Struct message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a Struct message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Struct} Struct
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Struct;

            /**
             * Creates a Struct message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Struct.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Struct} Struct
             */
            public static from(object: { [k: string]: any }): google.protobuf.Struct;

            /**
             * Creates a plain object from a Struct message. Also converts values to other types if specified.
             * @param {google.protobuf.Struct} message Struct
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: google.protobuf.Struct, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this Struct message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this Struct to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type Value$Properties = {
            nullValue?: google.protobuf.NullValue;
            numberValue?: number;
            stringValue?: string;
            boolValue?: boolean;
            structValue?: google.protobuf.Struct$Properties;
            listValue?: google.protobuf.ListValue$Properties;
        };

        /**
         * Constructs a new Value.
         * @exports google.protobuf.Value
         * @constructor
         * @param {google.protobuf.Value$Properties=} [properties] Properties to set
         */
        class Value {

            /**
             * Constructs a new Value.
             * @exports google.protobuf.Value
             * @constructor
             * @param {google.protobuf.Value$Properties=} [properties] Properties to set
             */
            constructor(properties?: google.protobuf.Value$Properties);

            /**
             * Value nullValue.
             * @type {google.protobuf.NullValue|undefined}
             */
            public nullValue?: google.protobuf.NullValue;

            /**
             * Value numberValue.
             * @type {number|undefined}
             */
            public numberValue?: number;

            /**
             * Value stringValue.
             * @type {string|undefined}
             */
            public stringValue?: string;

            /**
             * Value boolValue.
             * @type {boolean|undefined}
             */
            public boolValue?: boolean;

            /**
             * Value structValue.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            public structValue?: google.protobuf.Struct$Properties;

            /**
             * Value listValue.
             * @type {google.protobuf.ListValue$Properties|undefined}
             */
            public listValue?: google.protobuf.ListValue$Properties;

            /**
             * Value kind.
             * @name google.protobuf.Value#kind
             * @type {string|undefined}
             */
            public kind?: string;

            /**
             * Creates a new Value instance using the specified properties.
             * @param {google.protobuf.Value$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Value} Value instance
             */
            public static create(properties?: google.protobuf.Value$Properties): google.protobuf.Value;

            /**
             * Encodes the specified Value message. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param {google.protobuf.Value$Properties} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: google.protobuf.Value$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Value message, length delimited. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param {google.protobuf.Value$Properties} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: google.protobuf.Value$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Value message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Value} Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Value;

            /**
             * Decodes a Value message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Value} Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Value;

            /**
             * Verifies a Value message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Value} Value
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Value;

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Value.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Value} Value
             */
            public static from(object: { [k: string]: any }): google.protobuf.Value;

            /**
             * Creates a plain object from a Value message. Also converts values to other types if specified.
             * @param {google.protobuf.Value} message Value
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: google.protobuf.Value, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this Value message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this Value to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /**
         * NullValue enum.
         * @name NullValue
         * @memberof google.protobuf
         * @enum {number}
         * @property {number} NULL_VALUE=0 NULL_VALUE value
         */
        enum NullValue {
            NULL_VALUE = 0
        }

        type ListValue$Properties = {
            values?: google.protobuf.Value$Properties[];
        };

        /**
         * Constructs a new ListValue.
         * @exports google.protobuf.ListValue
         * @constructor
         * @param {google.protobuf.ListValue$Properties=} [properties] Properties to set
         */
        class ListValue {

            /**
             * Constructs a new ListValue.
             * @exports google.protobuf.ListValue
             * @constructor
             * @param {google.protobuf.ListValue$Properties=} [properties] Properties to set
             */
            constructor(properties?: google.protobuf.ListValue$Properties);

            /**
             * ListValue values.
             * @type {Array.<google.protobuf.Value$Properties>|undefined}
             */
            public values?: google.protobuf.Value$Properties[];

            /**
             * Creates a new ListValue instance using the specified properties.
             * @param {google.protobuf.ListValue$Properties=} [properties] Properties to set
             * @returns {google.protobuf.ListValue} ListValue instance
             */
            public static create(properties?: google.protobuf.ListValue$Properties): google.protobuf.ListValue;

            /**
             * Encodes the specified ListValue message. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param {google.protobuf.ListValue$Properties} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: google.protobuf.ListValue$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ListValue message, length delimited. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param {google.protobuf.ListValue$Properties} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: google.protobuf.ListValue$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ListValue message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.ListValue} ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.ListValue;

            /**
             * Decodes a ListValue message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.ListValue} ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.ListValue;

            /**
             * Verifies a ListValue message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.ListValue} ListValue
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.ListValue;

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.ListValue.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.ListValue} ListValue
             */
            public static from(object: { [k: string]: any }): google.protobuf.ListValue;

            /**
             * Creates a plain object from a ListValue message. Also converts values to other types if specified.
             * @param {google.protobuf.ListValue} message ListValue
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: google.protobuf.ListValue, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this ListValue message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this ListValue to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type Timestamp$Properties = {
            seconds?: (number|Long);
            nanos?: number;
        };

        /**
         * Constructs a new Timestamp.
         * @exports google.protobuf.Timestamp
         * @constructor
         * @param {google.protobuf.Timestamp$Properties=} [properties] Properties to set
         */
        class Timestamp {

            /**
             * Constructs a new Timestamp.
             * @exports google.protobuf.Timestamp
             * @constructor
             * @param {google.protobuf.Timestamp$Properties=} [properties] Properties to set
             */
            constructor(properties?: google.protobuf.Timestamp$Properties);

            /**
             * Timestamp seconds.
             * @type {number|Long|undefined}
             */
            public seconds?: (number|Long);

            /**
             * Timestamp nanos.
             * @type {number|undefined}
             */
            public nanos?: number;

            /**
             * Creates a new Timestamp instance using the specified properties.
             * @param {google.protobuf.Timestamp$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Timestamp} Timestamp instance
             */
            public static create(properties?: google.protobuf.Timestamp$Properties): google.protobuf.Timestamp;

            /**
             * Encodes the specified Timestamp message. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param {google.protobuf.Timestamp$Properties} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: google.protobuf.Timestamp$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Timestamp message, length delimited. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param {google.protobuf.Timestamp$Properties} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: google.protobuf.Timestamp$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Timestamp message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Timestamp} Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Timestamp;

            /**
             * Decodes a Timestamp message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Timestamp} Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Timestamp;

            /**
             * Verifies a Timestamp message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Timestamp} Timestamp
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Timestamp;

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Timestamp.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Timestamp} Timestamp
             */
            public static from(object: { [k: string]: any }): google.protobuf.Timestamp;

            /**
             * Creates a plain object from a Timestamp message. Also converts values to other types if specified.
             * @param {google.protobuf.Timestamp} message Timestamp
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: google.protobuf.Timestamp, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this Timestamp message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this Timestamp to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        type Empty$Properties = {};

        /**
         * Constructs a new Empty.
         * @exports google.protobuf.Empty
         * @constructor
         * @param {google.protobuf.Empty$Properties=} [properties] Properties to set
         */
        class Empty {

            /**
             * Constructs a new Empty.
             * @exports google.protobuf.Empty
             * @constructor
             * @param {google.protobuf.Empty$Properties=} [properties] Properties to set
             */
            constructor(properties?: google.protobuf.Empty$Properties);

            /**
             * Creates a new Empty instance using the specified properties.
             * @param {google.protobuf.Empty$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Empty} Empty instance
             */
            public static create(properties?: google.protobuf.Empty$Properties): google.protobuf.Empty;

            /**
             * Encodes the specified Empty message. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param {google.protobuf.Empty$Properties} message Empty message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encode(message: google.protobuf.Empty$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Empty message, length delimited. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param {google.protobuf.Empty$Properties} message Empty message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            public static encodeDelimited(message: google.protobuf.Empty$Properties, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an Empty message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Empty} Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): google.protobuf.Empty;

            /**
             * Decodes an Empty message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Empty} Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): google.protobuf.Empty;

            /**
             * Verifies an Empty message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): string;

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Empty} Empty
             */
            public static fromObject(object: { [k: string]: any }): google.protobuf.Empty;

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Empty.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Empty} Empty
             */
            public static from(object: { [k: string]: any }): google.protobuf.Empty;

            /**
             * Creates a plain object from an Empty message. Also converts values to other types if specified.
             * @param {google.protobuf.Empty} message Empty
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public static toObject(message: google.protobuf.Empty, options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Creates a plain object from this Empty message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            public toObject(options?: $protobuf.ConversionOptions): { [k: string]: any };

            /**
             * Converts this Empty to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }
}
