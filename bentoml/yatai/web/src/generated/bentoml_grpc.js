/*eslint-disable block-scoped-var, no-redeclare, no-control-regex, no-prototype-builtins*/
import * as $protobuf from "protobufjs";

// Common aliases
const $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;

// Exported root namespace
const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const bentoml = $root.bentoml = (() => {

    /**
     * Namespace bentoml.
     * @exports bentoml
     * @namespace
     */
    const bentoml = {};

    bentoml.DeploymentSpec = (function() {

        /**
         * Properties of a DeploymentSpec.
         * @typedef bentoml.DeploymentSpec$Properties
         * @type {Object}
         * @property {string} [bento_name] DeploymentSpec bento_name.
         * @property {string} [bento_version] DeploymentSpec bento_version.
         * @property {bentoml.DeploymentSpec.DeploymentOperator} [operator] DeploymentSpec operator.
         * @property {bentoml.DeploymentSpec.CustomOperatorConfig$Properties} [custom_operator_config] DeploymentSpec custom_operator_config.
         * @property {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties} [sagemaker_operator_config] DeploymentSpec sagemaker_operator_config.
         * @property {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties} [aws_lambda_operator_config] DeploymentSpec aws_lambda_operator_config.
         * @property {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties} [azure_functions_operator_config] DeploymentSpec azure_functions_operator_config.
         */

        /**
         * Constructs a new DeploymentSpec.
         * @exports bentoml.DeploymentSpec
         * @constructor
         * @param {bentoml.DeploymentSpec$Properties=} [properties] Properties to set
         */
        function DeploymentSpec(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentSpec bento_name.
         * @type {string|undefined}
         */
        DeploymentSpec.prototype.bento_name = "";

        /**
         * DeploymentSpec bento_version.
         * @type {string|undefined}
         */
        DeploymentSpec.prototype.bento_version = "";

        /**
         * DeploymentSpec operator.
         * @type {bentoml.DeploymentSpec.DeploymentOperator|undefined}
         */
        DeploymentSpec.prototype.operator = 0;

        /**
         * DeploymentSpec custom_operator_config.
         * @type {bentoml.DeploymentSpec.CustomOperatorConfig$Properties|undefined}
         */
        DeploymentSpec.prototype.custom_operator_config = null;

        /**
         * DeploymentSpec sagemaker_operator_config.
         * @type {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties|undefined}
         */
        DeploymentSpec.prototype.sagemaker_operator_config = null;

        /**
         * DeploymentSpec aws_lambda_operator_config.
         * @type {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties|undefined}
         */
        DeploymentSpec.prototype.aws_lambda_operator_config = null;

        /**
         * DeploymentSpec azure_functions_operator_config.
         * @type {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties|undefined}
         */
        DeploymentSpec.prototype.azure_functions_operator_config = null;

        // OneOf field names bound to virtual getters and setters
        let $oneOfFields;

        /**
         * DeploymentSpec deployment_operator_config.
         * @name bentoml.DeploymentSpec#deployment_operator_config
         * @type {string|undefined}
         */
        Object.defineProperty(DeploymentSpec.prototype, "deployment_operator_config", {
            get: $util.oneOfGetter($oneOfFields = ["custom_operator_config", "sagemaker_operator_config", "aws_lambda_operator_config", "azure_functions_operator_config"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        /**
         * Creates a new DeploymentSpec instance using the specified properties.
         * @param {bentoml.DeploymentSpec$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentSpec} DeploymentSpec instance
         */
        DeploymentSpec.create = function create(properties) {
            return new DeploymentSpec(properties);
        };

        /**
         * Encodes the specified DeploymentSpec message. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param {bentoml.DeploymentSpec$Properties} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentSpec.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            if (message.operator != null && message.hasOwnProperty("operator"))
                writer.uint32(/* id 3, wireType 0 =*/24).uint32(message.operator);
            if (message.custom_operator_config && message.hasOwnProperty("custom_operator_config"))
                $root.bentoml.DeploymentSpec.CustomOperatorConfig.encode(message.custom_operator_config, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.sagemaker_operator_config && message.hasOwnProperty("sagemaker_operator_config"))
                $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.encode(message.sagemaker_operator_config, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            if (message.aws_lambda_operator_config && message.hasOwnProperty("aws_lambda_operator_config"))
                $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.encode(message.aws_lambda_operator_config, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
            if (message.azure_functions_operator_config && message.hasOwnProperty("azure_functions_operator_config"))
                $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.encode(message.azure_functions_operator_config, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentSpec message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @param {bentoml.DeploymentSpec$Properties} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentSpec.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentSpec.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentSpec();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.bento_version = reader.string();
                    break;
                case 3:
                    message.operator = reader.uint32();
                    break;
                case 4:
                    message.custom_operator_config = $root.bentoml.DeploymentSpec.CustomOperatorConfig.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.sagemaker_operator_config = $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.aws_lambda_operator_config = $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.azure_functions_operator_config = $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentSpec.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentSpec message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DeploymentSpec.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            let properties = {};
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null)
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            if (message.operator != null)
                switch (message.operator) {
                default:
                    return "operator: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                    break;
                }
            if (message.custom_operator_config != null) {
                properties.deployment_operator_config = 1;
                let error = $root.bentoml.DeploymentSpec.CustomOperatorConfig.verify(message.custom_operator_config);
                if (error)
                    return "custom_operator_config." + error;
            }
            if (message.sagemaker_operator_config != null) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                let error = $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.verify(message.sagemaker_operator_config);
                if (error)
                    return "sagemaker_operator_config." + error;
            }
            if (message.aws_lambda_operator_config != null) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                let error = $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify(message.aws_lambda_operator_config);
                if (error)
                    return "aws_lambda_operator_config." + error;
            }
            if (message.azure_functions_operator_config != null) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                let error = $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify(message.azure_functions_operator_config);
                if (error)
                    return "azure_functions_operator_config." + error;
            }
            return null;
        };

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         */
        DeploymentSpec.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DeploymentSpec)
                return object;
            let message = new $root.bentoml.DeploymentSpec();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.bento_version != null)
                message.bento_version = String(object.bento_version);
            switch (object.operator) {
            case "UNSET":
            case 0:
                message.operator = 0;
                break;
            case "CUSTOM":
            case 1:
                message.operator = 1;
                break;
            case "AWS_SAGEMAKER":
            case 2:
                message.operator = 2;
                break;
            case "AWS_LAMBDA":
            case 3:
                message.operator = 3;
                break;
            case "AZURE_FUNCTIONS":
            case 4:
                message.operator = 4;
                break;
            }
            if (object.custom_operator_config != null) {
                if (typeof object.custom_operator_config !== "object")
                    throw TypeError(".bentoml.DeploymentSpec.custom_operator_config: object expected");
                message.custom_operator_config = $root.bentoml.DeploymentSpec.CustomOperatorConfig.fromObject(object.custom_operator_config);
            }
            if (object.sagemaker_operator_config != null) {
                if (typeof object.sagemaker_operator_config !== "object")
                    throw TypeError(".bentoml.DeploymentSpec.sagemaker_operator_config: object expected");
                message.sagemaker_operator_config = $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.fromObject(object.sagemaker_operator_config);
            }
            if (object.aws_lambda_operator_config != null) {
                if (typeof object.aws_lambda_operator_config !== "object")
                    throw TypeError(".bentoml.DeploymentSpec.aws_lambda_operator_config: object expected");
                message.aws_lambda_operator_config = $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.fromObject(object.aws_lambda_operator_config);
            }
            if (object.azure_functions_operator_config != null) {
                if (typeof object.azure_functions_operator_config !== "object")
                    throw TypeError(".bentoml.DeploymentSpec.azure_functions_operator_config: object expected");
                message.azure_functions_operator_config = $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.fromObject(object.azure_functions_operator_config);
            }
            return message;
        };

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentSpec.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         */
        DeploymentSpec.from = DeploymentSpec.fromObject;

        /**
         * Creates a plain object from a DeploymentSpec message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentSpec} message DeploymentSpec
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentSpec.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.bento_version = "";
                object.operator = options.enums === String ? "UNSET" : 0;
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                object.bento_version = message.bento_version;
            if (message.operator != null && message.hasOwnProperty("operator"))
                object.operator = options.enums === String ? $root.bentoml.DeploymentSpec.DeploymentOperator[message.operator] : message.operator;
            if (message.custom_operator_config != null && message.hasOwnProperty("custom_operator_config")) {
                object.custom_operator_config = $root.bentoml.DeploymentSpec.CustomOperatorConfig.toObject(message.custom_operator_config, options);
                if (options.oneofs)
                    object.deployment_operator_config = "custom_operator_config";
            }
            if (message.sagemaker_operator_config != null && message.hasOwnProperty("sagemaker_operator_config")) {
                object.sagemaker_operator_config = $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.toObject(message.sagemaker_operator_config, options);
                if (options.oneofs)
                    object.deployment_operator_config = "sagemaker_operator_config";
            }
            if (message.aws_lambda_operator_config != null && message.hasOwnProperty("aws_lambda_operator_config")) {
                object.aws_lambda_operator_config = $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.toObject(message.aws_lambda_operator_config, options);
                if (options.oneofs)
                    object.deployment_operator_config = "aws_lambda_operator_config";
            }
            if (message.azure_functions_operator_config != null && message.hasOwnProperty("azure_functions_operator_config")) {
                object.azure_functions_operator_config = $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.toObject(message.azure_functions_operator_config, options);
                if (options.oneofs)
                    object.deployment_operator_config = "azure_functions_operator_config";
            }
            return object;
        };

        /**
         * Creates a plain object from this DeploymentSpec message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentSpec.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DeploymentSpec to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DeploymentSpec.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

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
        DeploymentSpec.DeploymentOperator = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "UNSET"] = 0;
            values[valuesById[1] = "CUSTOM"] = 1;
            values[valuesById[2] = "AWS_SAGEMAKER"] = 2;
            values[valuesById[3] = "AWS_LAMBDA"] = 3;
            values[valuesById[4] = "AZURE_FUNCTIONS"] = 4;
            return values;
        })();

        DeploymentSpec.CustomOperatorConfig = (function() {

            /**
             * Properties of a CustomOperatorConfig.
             * @typedef bentoml.DeploymentSpec.CustomOperatorConfig$Properties
             * @type {Object}
             * @property {string} [name] CustomOperatorConfig name.
             * @property {google.protobuf.Struct$Properties} [config] CustomOperatorConfig config.
             */

            /**
             * Constructs a new CustomOperatorConfig.
             * @exports bentoml.DeploymentSpec.CustomOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties=} [properties] Properties to set
             */
            function CustomOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * CustomOperatorConfig name.
             * @type {string|undefined}
             */
            CustomOperatorConfig.prototype.name = "";

            /**
             * CustomOperatorConfig config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            CustomOperatorConfig.prototype.config = null;

            /**
             * Creates a new CustomOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig instance
             */
            CustomOperatorConfig.create = function create(properties) {
                return new CustomOperatorConfig(properties);
            };

            /**
             * Encodes the specified CustomOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CustomOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.config && message.hasOwnProperty("config"))
                    $root.google.protobuf.Struct.encode(message.config, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified CustomOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig$Properties} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CustomOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CustomOperatorConfig.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentSpec.CustomOperatorConfig();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.config = $root.google.protobuf.Struct.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CustomOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a CustomOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            CustomOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null)
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.config != null) {
                    let error = $root.google.protobuf.Struct.verify(message.config);
                    if (error)
                        return "config." + error;
                }
                return null;
            };

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             */
            CustomOperatorConfig.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.DeploymentSpec.CustomOperatorConfig)
                    return object;
                let message = new $root.bentoml.DeploymentSpec.CustomOperatorConfig();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.config != null) {
                    if (typeof object.config !== "object")
                        throw TypeError(".bentoml.DeploymentSpec.CustomOperatorConfig.config: object expected");
                    message.config = $root.google.protobuf.Struct.fromObject(object.config);
                }
                return message;
            };

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.CustomOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             */
            CustomOperatorConfig.from = CustomOperatorConfig.fromObject;

            /**
             * Creates a plain object from a CustomOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig} message CustomOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            CustomOperatorConfig.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.name = "";
                    object.config = null;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.config != null && message.hasOwnProperty("config"))
                    object.config = $root.google.protobuf.Struct.toObject(message.config, options);
                return object;
            };

            /**
             * Creates a plain object from this CustomOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            CustomOperatorConfig.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this CustomOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            CustomOperatorConfig.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return CustomOperatorConfig;
        })();

        DeploymentSpec.SageMakerOperatorConfig = (function() {

            /**
             * Properties of a SageMakerOperatorConfig.
             * @typedef bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties
             * @type {Object}
             * @property {string} [region] SageMakerOperatorConfig region.
             * @property {string} [instance_type] SageMakerOperatorConfig instance_type.
             * @property {number} [instance_count] SageMakerOperatorConfig instance_count.
             * @property {string} [api_name] SageMakerOperatorConfig api_name.
             * @property {number} [num_of_gunicorn_workers_per_instance] SageMakerOperatorConfig num_of_gunicorn_workers_per_instance.
             * @property {number} [timeout] SageMakerOperatorConfig timeout.
             */

            /**
             * Constructs a new SageMakerOperatorConfig.
             * @exports bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties=} [properties] Properties to set
             */
            function SageMakerOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * SageMakerOperatorConfig region.
             * @type {string|undefined}
             */
            SageMakerOperatorConfig.prototype.region = "";

            /**
             * SageMakerOperatorConfig instance_type.
             * @type {string|undefined}
             */
            SageMakerOperatorConfig.prototype.instance_type = "";

            /**
             * SageMakerOperatorConfig instance_count.
             * @type {number|undefined}
             */
            SageMakerOperatorConfig.prototype.instance_count = 0;

            /**
             * SageMakerOperatorConfig api_name.
             * @type {string|undefined}
             */
            SageMakerOperatorConfig.prototype.api_name = "";

            /**
             * SageMakerOperatorConfig num_of_gunicorn_workers_per_instance.
             * @type {number|undefined}
             */
            SageMakerOperatorConfig.prototype.num_of_gunicorn_workers_per_instance = 0;

            /**
             * SageMakerOperatorConfig timeout.
             * @type {number|undefined}
             */
            SageMakerOperatorConfig.prototype.timeout = 0;

            /**
             * Creates a new SageMakerOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig instance
             */
            SageMakerOperatorConfig.create = function create(properties) {
                return new SageMakerOperatorConfig(properties);
            };

            /**
             * Encodes the specified SageMakerOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SageMakerOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.region != null && message.hasOwnProperty("region"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.region);
                if (message.instance_type != null && message.hasOwnProperty("instance_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.instance_type);
                if (message.instance_count != null && message.hasOwnProperty("instance_count"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.instance_count);
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.api_name);
                if (message.num_of_gunicorn_workers_per_instance != null && message.hasOwnProperty("num_of_gunicorn_workers_per_instance"))
                    writer.uint32(/* id 5, wireType 0 =*/40).int32(message.num_of_gunicorn_workers_per_instance);
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    writer.uint32(/* id 6, wireType 0 =*/48).int32(message.timeout);
                return writer;
            };

            /**
             * Encodes the specified SageMakerOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig$Properties} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SageMakerOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SageMakerOperatorConfig.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentSpec.SageMakerOperatorConfig();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.region = reader.string();
                        break;
                    case 2:
                        message.instance_type = reader.string();
                        break;
                    case 3:
                        message.instance_count = reader.int32();
                        break;
                    case 4:
                        message.api_name = reader.string();
                        break;
                    case 5:
                        message.num_of_gunicorn_workers_per_instance = reader.int32();
                        break;
                    case 6:
                        message.timeout = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SageMakerOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a SageMakerOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            SageMakerOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.region != null)
                    if (!$util.isString(message.region))
                        return "region: string expected";
                if (message.instance_type != null)
                    if (!$util.isString(message.instance_type))
                        return "instance_type: string expected";
                if (message.instance_count != null)
                    if (!$util.isInteger(message.instance_count))
                        return "instance_count: integer expected";
                if (message.api_name != null)
                    if (!$util.isString(message.api_name))
                        return "api_name: string expected";
                if (message.num_of_gunicorn_workers_per_instance != null)
                    if (!$util.isInteger(message.num_of_gunicorn_workers_per_instance))
                        return "num_of_gunicorn_workers_per_instance: integer expected";
                if (message.timeout != null)
                    if (!$util.isInteger(message.timeout))
                        return "timeout: integer expected";
                return null;
            };

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             */
            SageMakerOperatorConfig.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.DeploymentSpec.SageMakerOperatorConfig)
                    return object;
                let message = new $root.bentoml.DeploymentSpec.SageMakerOperatorConfig();
                if (object.region != null)
                    message.region = String(object.region);
                if (object.instance_type != null)
                    message.instance_type = String(object.instance_type);
                if (object.instance_count != null)
                    message.instance_count = object.instance_count | 0;
                if (object.api_name != null)
                    message.api_name = String(object.api_name);
                if (object.num_of_gunicorn_workers_per_instance != null)
                    message.num_of_gunicorn_workers_per_instance = object.num_of_gunicorn_workers_per_instance | 0;
                if (object.timeout != null)
                    message.timeout = object.timeout | 0;
                return message;
            };

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             */
            SageMakerOperatorConfig.from = SageMakerOperatorConfig.fromObject;

            /**
             * Creates a plain object from a SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig} message SageMakerOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            SageMakerOperatorConfig.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.region = "";
                    object.instance_type = "";
                    object.instance_count = 0;
                    object.api_name = "";
                    object.num_of_gunicorn_workers_per_instance = 0;
                    object.timeout = 0;
                }
                if (message.region != null && message.hasOwnProperty("region"))
                    object.region = message.region;
                if (message.instance_type != null && message.hasOwnProperty("instance_type"))
                    object.instance_type = message.instance_type;
                if (message.instance_count != null && message.hasOwnProperty("instance_count"))
                    object.instance_count = message.instance_count;
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    object.api_name = message.api_name;
                if (message.num_of_gunicorn_workers_per_instance != null && message.hasOwnProperty("num_of_gunicorn_workers_per_instance"))
                    object.num_of_gunicorn_workers_per_instance = message.num_of_gunicorn_workers_per_instance;
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    object.timeout = message.timeout;
                return object;
            };

            /**
             * Creates a plain object from this SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            SageMakerOperatorConfig.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this SageMakerOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            SageMakerOperatorConfig.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return SageMakerOperatorConfig;
        })();

        DeploymentSpec.AwsLambdaOperatorConfig = (function() {

            /**
             * Properties of an AwsLambdaOperatorConfig.
             * @typedef bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties
             * @type {Object}
             * @property {string} [region] AwsLambdaOperatorConfig region.
             * @property {string} [api_name] AwsLambdaOperatorConfig api_name.
             * @property {number} [memory_size] AwsLambdaOperatorConfig memory_size.
             * @property {number} [timeout] AwsLambdaOperatorConfig timeout.
             */

            /**
             * Constructs a new AwsLambdaOperatorConfig.
             * @exports bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties=} [properties] Properties to set
             */
            function AwsLambdaOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * AwsLambdaOperatorConfig region.
             * @type {string|undefined}
             */
            AwsLambdaOperatorConfig.prototype.region = "";

            /**
             * AwsLambdaOperatorConfig api_name.
             * @type {string|undefined}
             */
            AwsLambdaOperatorConfig.prototype.api_name = "";

            /**
             * AwsLambdaOperatorConfig memory_size.
             * @type {number|undefined}
             */
            AwsLambdaOperatorConfig.prototype.memory_size = 0;

            /**
             * AwsLambdaOperatorConfig timeout.
             * @type {number|undefined}
             */
            AwsLambdaOperatorConfig.prototype.timeout = 0;

            /**
             * Creates a new AwsLambdaOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig instance
             */
            AwsLambdaOperatorConfig.create = function create(properties) {
                return new AwsLambdaOperatorConfig(properties);
            };

            /**
             * Encodes the specified AwsLambdaOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AwsLambdaOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.region != null && message.hasOwnProperty("region"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.region);
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.api_name);
                if (message.memory_size != null && message.hasOwnProperty("memory_size"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.memory_size);
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.timeout);
                return writer;
            };

            /**
             * Encodes the specified AwsLambdaOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig$Properties} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AwsLambdaOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AwsLambdaOperatorConfig.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.region = reader.string();
                        break;
                    case 2:
                        message.api_name = reader.string();
                        break;
                    case 3:
                        message.memory_size = reader.int32();
                        break;
                    case 4:
                        message.timeout = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AwsLambdaOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AwsLambdaOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            AwsLambdaOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.region != null)
                    if (!$util.isString(message.region))
                        return "region: string expected";
                if (message.api_name != null)
                    if (!$util.isString(message.api_name))
                        return "api_name: string expected";
                if (message.memory_size != null)
                    if (!$util.isInteger(message.memory_size))
                        return "memory_size: integer expected";
                if (message.timeout != null)
                    if (!$util.isInteger(message.timeout))
                        return "timeout: integer expected";
                return null;
            };

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             */
            AwsLambdaOperatorConfig.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig)
                    return object;
                let message = new $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig();
                if (object.region != null)
                    message.region = String(object.region);
                if (object.api_name != null)
                    message.api_name = String(object.api_name);
                if (object.memory_size != null)
                    message.memory_size = object.memory_size | 0;
                if (object.timeout != null)
                    message.timeout = object.timeout | 0;
                return message;
            };

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             */
            AwsLambdaOperatorConfig.from = AwsLambdaOperatorConfig.fromObject;

            /**
             * Creates a plain object from an AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} message AwsLambdaOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AwsLambdaOperatorConfig.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.region = "";
                    object.api_name = "";
                    object.memory_size = 0;
                    object.timeout = 0;
                }
                if (message.region != null && message.hasOwnProperty("region"))
                    object.region = message.region;
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    object.api_name = message.api_name;
                if (message.memory_size != null && message.hasOwnProperty("memory_size"))
                    object.memory_size = message.memory_size;
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    object.timeout = message.timeout;
                return object;
            };

            /**
             * Creates a plain object from this AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AwsLambdaOperatorConfig.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this AwsLambdaOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            AwsLambdaOperatorConfig.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return AwsLambdaOperatorConfig;
        })();

        DeploymentSpec.AzureFunctionsOperatorConfig = (function() {

            /**
             * Properties of an AzureFunctionsOperatorConfig.
             * @typedef bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties
             * @type {Object}
             * @property {string} [location] AzureFunctionsOperatorConfig location.
             * @property {string} [premium_plan_sku] AzureFunctionsOperatorConfig premium_plan_sku.
             * @property {number} [min_instances] AzureFunctionsOperatorConfig min_instances.
             * @property {number} [max_burst] AzureFunctionsOperatorConfig max_burst.
             * @property {string} [function_auth_level] AzureFunctionsOperatorConfig function_auth_level.
             */

            /**
             * Constructs a new AzureFunctionsOperatorConfig.
             * @exports bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties=} [properties] Properties to set
             */
            function AzureFunctionsOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * AzureFunctionsOperatorConfig location.
             * @type {string|undefined}
             */
            AzureFunctionsOperatorConfig.prototype.location = "";

            /**
             * AzureFunctionsOperatorConfig premium_plan_sku.
             * @type {string|undefined}
             */
            AzureFunctionsOperatorConfig.prototype.premium_plan_sku = "";

            /**
             * AzureFunctionsOperatorConfig min_instances.
             * @type {number|undefined}
             */
            AzureFunctionsOperatorConfig.prototype.min_instances = 0;

            /**
             * AzureFunctionsOperatorConfig max_burst.
             * @type {number|undefined}
             */
            AzureFunctionsOperatorConfig.prototype.max_burst = 0;

            /**
             * AzureFunctionsOperatorConfig function_auth_level.
             * @type {string|undefined}
             */
            AzureFunctionsOperatorConfig.prototype.function_auth_level = "";

            /**
             * Creates a new AzureFunctionsOperatorConfig instance using the specified properties.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig instance
             */
            AzureFunctionsOperatorConfig.create = function create(properties) {
                return new AzureFunctionsOperatorConfig(properties);
            };

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AzureFunctionsOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.location != null && message.hasOwnProperty("location"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.location);
                if (message.premium_plan_sku != null && message.hasOwnProperty("premium_plan_sku"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.premium_plan_sku);
                if (message.min_instances != null && message.hasOwnProperty("min_instances"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.min_instances);
                if (message.max_burst != null && message.hasOwnProperty("max_burst"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.max_burst);
                if (message.function_auth_level != null && message.hasOwnProperty("function_auth_level"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.function_auth_level);
                return writer;
            };

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig$Properties} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AzureFunctionsOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AzureFunctionsOperatorConfig.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.location = reader.string();
                        break;
                    case 2:
                        message.premium_plan_sku = reader.string();
                        break;
                    case 3:
                        message.min_instances = reader.int32();
                        break;
                    case 4:
                        message.max_burst = reader.int32();
                        break;
                    case 5:
                        message.function_auth_level = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AzureFunctionsOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AzureFunctionsOperatorConfig message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            AzureFunctionsOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.location != null)
                    if (!$util.isString(message.location))
                        return "location: string expected";
                if (message.premium_plan_sku != null)
                    if (!$util.isString(message.premium_plan_sku))
                        return "premium_plan_sku: string expected";
                if (message.min_instances != null)
                    if (!$util.isInteger(message.min_instances))
                        return "min_instances: integer expected";
                if (message.max_burst != null)
                    if (!$util.isInteger(message.max_burst))
                        return "max_burst: integer expected";
                if (message.function_auth_level != null)
                    if (!$util.isString(message.function_auth_level))
                        return "function_auth_level: string expected";
                return null;
            };

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             */
            AzureFunctionsOperatorConfig.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig)
                    return object;
                let message = new $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig();
                if (object.location != null)
                    message.location = String(object.location);
                if (object.premium_plan_sku != null)
                    message.premium_plan_sku = String(object.premium_plan_sku);
                if (object.min_instances != null)
                    message.min_instances = object.min_instances | 0;
                if (object.max_burst != null)
                    message.max_burst = object.max_burst | 0;
                if (object.function_auth_level != null)
                    message.function_auth_level = String(object.function_auth_level);
                return message;
            };

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             */
            AzureFunctionsOperatorConfig.from = AzureFunctionsOperatorConfig.fromObject;

            /**
             * Creates a plain object from an AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} message AzureFunctionsOperatorConfig
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AzureFunctionsOperatorConfig.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.location = "";
                    object.premium_plan_sku = "";
                    object.min_instances = 0;
                    object.max_burst = 0;
                    object.function_auth_level = "";
                }
                if (message.location != null && message.hasOwnProperty("location"))
                    object.location = message.location;
                if (message.premium_plan_sku != null && message.hasOwnProperty("premium_plan_sku"))
                    object.premium_plan_sku = message.premium_plan_sku;
                if (message.min_instances != null && message.hasOwnProperty("min_instances"))
                    object.min_instances = message.min_instances;
                if (message.max_burst != null && message.hasOwnProperty("max_burst"))
                    object.max_burst = message.max_burst;
                if (message.function_auth_level != null && message.hasOwnProperty("function_auth_level"))
                    object.function_auth_level = message.function_auth_level;
                return object;
            };

            /**
             * Creates a plain object from this AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AzureFunctionsOperatorConfig.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this AzureFunctionsOperatorConfig to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            AzureFunctionsOperatorConfig.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return AzureFunctionsOperatorConfig;
        })();

        return DeploymentSpec;
    })();

    bentoml.DeploymentState = (function() {

        /**
         * Properties of a DeploymentState.
         * @typedef bentoml.DeploymentState$Properties
         * @type {Object}
         * @property {bentoml.DeploymentState.State} [state] DeploymentState state.
         * @property {string} [error_message] DeploymentState error_message.
         * @property {string} [info_json] DeploymentState info_json.
         * @property {google.protobuf.Timestamp$Properties} [timestamp] DeploymentState timestamp.
         */

        /**
         * Constructs a new DeploymentState.
         * @exports bentoml.DeploymentState
         * @constructor
         * @param {bentoml.DeploymentState$Properties=} [properties] Properties to set
         */
        function DeploymentState(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentState state.
         * @type {bentoml.DeploymentState.State|undefined}
         */
        DeploymentState.prototype.state = 0;

        /**
         * DeploymentState error_message.
         * @type {string|undefined}
         */
        DeploymentState.prototype.error_message = "";

        /**
         * DeploymentState info_json.
         * @type {string|undefined}
         */
        DeploymentState.prototype.info_json = "";

        /**
         * DeploymentState timestamp.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        DeploymentState.prototype.timestamp = null;

        /**
         * Creates a new DeploymentState instance using the specified properties.
         * @param {bentoml.DeploymentState$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentState} DeploymentState instance
         */
        DeploymentState.create = function create(properties) {
            return new DeploymentState(properties);
        };

        /**
         * Encodes the specified DeploymentState message. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param {bentoml.DeploymentState$Properties} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentState.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.state != null && message.hasOwnProperty("state"))
                writer.uint32(/* id 1, wireType 0 =*/8).uint32(message.state);
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.error_message);
            if (message.info_json != null && message.hasOwnProperty("info_json"))
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.info_json);
            if (message.timestamp && message.hasOwnProperty("timestamp"))
                $root.google.protobuf.Timestamp.encode(message.timestamp, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentState message, length delimited. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @param {bentoml.DeploymentState$Properties} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentState.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentState message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentState} DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentState.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentState();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.state = reader.uint32();
                    break;
                case 2:
                    message.error_message = reader.string();
                    break;
                case 3:
                    message.info_json = reader.string();
                    break;
                case 4:
                    message.timestamp = $root.google.protobuf.Timestamp.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DeploymentState message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentState} DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentState.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentState message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DeploymentState.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.state != null)
                switch (message.state) {
                default:
                    return "state: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                    break;
                }
            if (message.error_message != null)
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            if (message.info_json != null)
                if (!$util.isString(message.info_json))
                    return "info_json: string expected";
            if (message.timestamp != null) {
                let error = $root.google.protobuf.Timestamp.verify(message.timestamp);
                if (error)
                    return "timestamp." + error;
            }
            return null;
        };

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentState} DeploymentState
         */
        DeploymentState.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DeploymentState)
                return object;
            let message = new $root.bentoml.DeploymentState();
            switch (object.state) {
            case "PENDING":
            case 0:
                message.state = 0;
                break;
            case "RUNNING":
            case 1:
                message.state = 1;
                break;
            case "SUCCEEDED":
            case 2:
                message.state = 2;
                break;
            case "FAILED":
            case 3:
                message.state = 3;
                break;
            case "UNKNOWN":
            case 4:
                message.state = 4;
                break;
            case "COMPLETED":
            case 5:
                message.state = 5;
                break;
            case "CRASH_LOOP_BACK_OFF":
            case 6:
                message.state = 6;
                break;
            case "ERROR":
            case 7:
                message.state = 7;
                break;
            case "INACTIVATED":
            case 8:
                message.state = 8;
                break;
            }
            if (object.error_message != null)
                message.error_message = String(object.error_message);
            if (object.info_json != null)
                message.info_json = String(object.info_json);
            if (object.timestamp != null) {
                if (typeof object.timestamp !== "object")
                    throw TypeError(".bentoml.DeploymentState.timestamp: object expected");
                message.timestamp = $root.google.protobuf.Timestamp.fromObject(object.timestamp);
            }
            return message;
        };

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentState.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentState} DeploymentState
         */
        DeploymentState.from = DeploymentState.fromObject;

        /**
         * Creates a plain object from a DeploymentState message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentState} message DeploymentState
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentState.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.state = options.enums === String ? "PENDING" : 0;
                object.error_message = "";
                object.info_json = "";
                object.timestamp = null;
            }
            if (message.state != null && message.hasOwnProperty("state"))
                object.state = options.enums === String ? $root.bentoml.DeploymentState.State[message.state] : message.state;
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                object.error_message = message.error_message;
            if (message.info_json != null && message.hasOwnProperty("info_json"))
                object.info_json = message.info_json;
            if (message.timestamp != null && message.hasOwnProperty("timestamp"))
                object.timestamp = $root.google.protobuf.Timestamp.toObject(message.timestamp, options);
            return object;
        };

        /**
         * Creates a plain object from this DeploymentState message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentState.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DeploymentState to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DeploymentState.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

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
        DeploymentState.State = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "PENDING"] = 0;
            values[valuesById[1] = "RUNNING"] = 1;
            values[valuesById[2] = "SUCCEEDED"] = 2;
            values[valuesById[3] = "FAILED"] = 3;
            values[valuesById[4] = "UNKNOWN"] = 4;
            values[valuesById[5] = "COMPLETED"] = 5;
            values[valuesById[6] = "CRASH_LOOP_BACK_OFF"] = 6;
            values[valuesById[7] = "ERROR"] = 7;
            values[valuesById[8] = "INACTIVATED"] = 8;
            return values;
        })();

        return DeploymentState;
    })();

    bentoml.Deployment = (function() {

        /**
         * Properties of a Deployment.
         * @typedef bentoml.Deployment$Properties
         * @type {Object}
         * @property {string} [namespace] Deployment namespace.
         * @property {string} [name] Deployment name.
         * @property {bentoml.DeploymentSpec$Properties} [spec] Deployment spec.
         * @property {bentoml.DeploymentState$Properties} [state] Deployment state.
         * @property {Object.<string,string>} [annotations] Deployment annotations.
         * @property {Object.<string,string>} [labels] Deployment labels.
         * @property {google.protobuf.Timestamp$Properties} [created_at] Deployment created_at.
         * @property {google.protobuf.Timestamp$Properties} [last_updated_at] Deployment last_updated_at.
         */

        /**
         * Constructs a new Deployment.
         * @exports bentoml.Deployment
         * @constructor
         * @param {bentoml.Deployment$Properties=} [properties] Properties to set
         */
        function Deployment(properties) {
            this.annotations = {};
            this.labels = {};
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * Deployment namespace.
         * @type {string|undefined}
         */
        Deployment.prototype.namespace = "";

        /**
         * Deployment name.
         * @type {string|undefined}
         */
        Deployment.prototype.name = "";

        /**
         * Deployment spec.
         * @type {bentoml.DeploymentSpec$Properties|undefined}
         */
        Deployment.prototype.spec = null;

        /**
         * Deployment state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        Deployment.prototype.state = null;

        /**
         * Deployment annotations.
         * @type {Object.<string,string>|undefined}
         */
        Deployment.prototype.annotations = $util.emptyObject;

        /**
         * Deployment labels.
         * @type {Object.<string,string>|undefined}
         */
        Deployment.prototype.labels = $util.emptyObject;

        /**
         * Deployment created_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        Deployment.prototype.created_at = null;

        /**
         * Deployment last_updated_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        Deployment.prototype.last_updated_at = null;

        /**
         * Creates a new Deployment instance using the specified properties.
         * @param {bentoml.Deployment$Properties=} [properties] Properties to set
         * @returns {bentoml.Deployment} Deployment instance
         */
        Deployment.create = function create(properties) {
            return new Deployment(properties);
        };

        /**
         * Encodes the specified Deployment message. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param {bentoml.Deployment$Properties} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Deployment.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.namespace);
            if (message.name != null && message.hasOwnProperty("name"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.name);
            if (message.spec && message.hasOwnProperty("spec"))
                $root.bentoml.DeploymentSpec.encode(message.spec, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.state && message.hasOwnProperty("state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.annotations && message.hasOwnProperty("annotations"))
                for (let keys = Object.keys(message.annotations), i = 0; i < keys.length; ++i)
                    writer.uint32(/* id 5, wireType 2 =*/42).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.annotations[keys[i]]).ldelim();
            if (message.labels && message.hasOwnProperty("labels"))
                for (let keys = Object.keys(message.labels), i = 0; i < keys.length; ++i)
                    writer.uint32(/* id 6, wireType 2 =*/50).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.labels[keys[i]]).ldelim();
            if (message.created_at && message.hasOwnProperty("created_at"))
                $root.google.protobuf.Timestamp.encode(message.created_at, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
            if (message.last_updated_at && message.hasOwnProperty("last_updated_at"))
                $root.google.protobuf.Timestamp.encode(message.last_updated_at, writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified Deployment message, length delimited. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @param {bentoml.Deployment$Properties} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Deployment.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Deployment message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Deployment} Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Deployment.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.Deployment(), key;
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.namespace = reader.string();
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.spec = $root.bentoml.DeploymentSpec.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.state = $root.bentoml.DeploymentState.decode(reader, reader.uint32());
                    break;
                case 5:
                    reader.skip().pos++;
                    if (message.annotations === $util.emptyObject)
                        message.annotations = {};
                    key = reader.string();
                    reader.pos++;
                    message.annotations[key] = reader.string();
                    break;
                case 6:
                    reader.skip().pos++;
                    if (message.labels === $util.emptyObject)
                        message.labels = {};
                    key = reader.string();
                    reader.pos++;
                    message.labels[key] = reader.string();
                    break;
                case 7:
                    message.created_at = $root.google.protobuf.Timestamp.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.last_updated_at = $root.google.protobuf.Timestamp.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Deployment message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Deployment} Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Deployment.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Deployment message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        Deployment.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.namespace != null)
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.name != null)
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.spec != null) {
                let error = $root.bentoml.DeploymentSpec.verify(message.spec);
                if (error)
                    return "spec." + error;
            }
            if (message.state != null) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            if (message.annotations != null) {
                if (!$util.isObject(message.annotations))
                    return "annotations: object expected";
                let key = Object.keys(message.annotations);
                for (let i = 0; i < key.length; ++i)
                    if (!$util.isString(message.annotations[key[i]]))
                        return "annotations: string{k:string} expected";
            }
            if (message.labels != null) {
                if (!$util.isObject(message.labels))
                    return "labels: object expected";
                let key = Object.keys(message.labels);
                for (let i = 0; i < key.length; ++i)
                    if (!$util.isString(message.labels[key[i]]))
                        return "labels: string{k:string} expected";
            }
            if (message.created_at != null) {
                let error = $root.google.protobuf.Timestamp.verify(message.created_at);
                if (error)
                    return "created_at." + error;
            }
            if (message.last_updated_at != null) {
                let error = $root.google.protobuf.Timestamp.verify(message.last_updated_at);
                if (error)
                    return "last_updated_at." + error;
            }
            return null;
        };

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Deployment} Deployment
         */
        Deployment.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.Deployment)
                return object;
            let message = new $root.bentoml.Deployment();
            if (object.namespace != null)
                message.namespace = String(object.namespace);
            if (object.name != null)
                message.name = String(object.name);
            if (object.spec != null) {
                if (typeof object.spec !== "object")
                    throw TypeError(".bentoml.Deployment.spec: object expected");
                message.spec = $root.bentoml.DeploymentSpec.fromObject(object.spec);
            }
            if (object.state != null) {
                if (typeof object.state !== "object")
                    throw TypeError(".bentoml.Deployment.state: object expected");
                message.state = $root.bentoml.DeploymentState.fromObject(object.state);
            }
            if (object.annotations) {
                if (typeof object.annotations !== "object")
                    throw TypeError(".bentoml.Deployment.annotations: object expected");
                message.annotations = {};
                for (let keys = Object.keys(object.annotations), i = 0; i < keys.length; ++i)
                    message.annotations[keys[i]] = String(object.annotations[keys[i]]);
            }
            if (object.labels) {
                if (typeof object.labels !== "object")
                    throw TypeError(".bentoml.Deployment.labels: object expected");
                message.labels = {};
                for (let keys = Object.keys(object.labels), i = 0; i < keys.length; ++i)
                    message.labels[keys[i]] = String(object.labels[keys[i]]);
            }
            if (object.created_at != null) {
                if (typeof object.created_at !== "object")
                    throw TypeError(".bentoml.Deployment.created_at: object expected");
                message.created_at = $root.google.protobuf.Timestamp.fromObject(object.created_at);
            }
            if (object.last_updated_at != null) {
                if (typeof object.last_updated_at !== "object")
                    throw TypeError(".bentoml.Deployment.last_updated_at: object expected");
                message.last_updated_at = $root.google.protobuf.Timestamp.fromObject(object.last_updated_at);
            }
            return message;
        };

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Deployment.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Deployment} Deployment
         */
        Deployment.from = Deployment.fromObject;

        /**
         * Creates a plain object from a Deployment message. Also converts values to other types if specified.
         * @param {bentoml.Deployment} message Deployment
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Deployment.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.objects || options.defaults) {
                object.annotations = {};
                object.labels = {};
            }
            if (options.defaults) {
                object.namespace = "";
                object.name = "";
                object.spec = null;
                object.state = null;
                object.created_at = null;
                object.last_updated_at = null;
            }
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                object.namespace = message.namespace;
            if (message.name != null && message.hasOwnProperty("name"))
                object.name = message.name;
            if (message.spec != null && message.hasOwnProperty("spec"))
                object.spec = $root.bentoml.DeploymentSpec.toObject(message.spec, options);
            if (message.state != null && message.hasOwnProperty("state"))
                object.state = $root.bentoml.DeploymentState.toObject(message.state, options);
            let keys2;
            if (message.annotations && (keys2 = Object.keys(message.annotations)).length) {
                object.annotations = {};
                for (let j = 0; j < keys2.length; ++j)
                    object.annotations[keys2[j]] = message.annotations[keys2[j]];
            }
            if (message.labels && (keys2 = Object.keys(message.labels)).length) {
                object.labels = {};
                for (let j = 0; j < keys2.length; ++j)
                    object.labels[keys2[j]] = message.labels[keys2[j]];
            }
            if (message.created_at != null && message.hasOwnProperty("created_at"))
                object.created_at = $root.google.protobuf.Timestamp.toObject(message.created_at, options);
            if (message.last_updated_at != null && message.hasOwnProperty("last_updated_at"))
                object.last_updated_at = $root.google.protobuf.Timestamp.toObject(message.last_updated_at, options);
            return object;
        };

        /**
         * Creates a plain object from this Deployment message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Deployment.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this Deployment to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        Deployment.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Deployment;
    })();

    bentoml.DeploymentStatus = (function() {

        /**
         * Properties of a DeploymentStatus.
         * @typedef bentoml.DeploymentStatus$Properties
         * @type {Object}
         * @property {bentoml.DeploymentState$Properties} [state] DeploymentStatus state.
         */

        /**
         * Constructs a new DeploymentStatus.
         * @exports bentoml.DeploymentStatus
         * @constructor
         * @param {bentoml.DeploymentStatus$Properties=} [properties] Properties to set
         */
        function DeploymentStatus(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentStatus state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        DeploymentStatus.prototype.state = null;

        /**
         * Creates a new DeploymentStatus instance using the specified properties.
         * @param {bentoml.DeploymentStatus$Properties=} [properties] Properties to set
         * @returns {bentoml.DeploymentStatus} DeploymentStatus instance
         */
        DeploymentStatus.create = function create(properties) {
            return new DeploymentStatus(properties);
        };

        /**
         * Encodes the specified DeploymentStatus message. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param {bentoml.DeploymentStatus$Properties} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentStatus.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.state && message.hasOwnProperty("state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentStatus message, length delimited. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @param {bentoml.DeploymentStatus$Properties} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentStatus.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentStatus.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeploymentStatus();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.state = $root.bentoml.DeploymentState.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentStatus.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentStatus message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DeploymentStatus.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.state != null) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            return null;
        };

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         */
        DeploymentStatus.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DeploymentStatus)
                return object;
            let message = new $root.bentoml.DeploymentStatus();
            if (object.state != null) {
                if (typeof object.state !== "object")
                    throw TypeError(".bentoml.DeploymentStatus.state: object expected");
                message.state = $root.bentoml.DeploymentState.fromObject(object.state);
            }
            return message;
        };

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeploymentStatus.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         */
        DeploymentStatus.from = DeploymentStatus.fromObject;

        /**
         * Creates a plain object from a DeploymentStatus message. Also converts values to other types if specified.
         * @param {bentoml.DeploymentStatus} message DeploymentStatus
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentStatus.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.state = null;
            if (message.state != null && message.hasOwnProperty("state"))
                object.state = $root.bentoml.DeploymentState.toObject(message.state, options);
            return object;
        };

        /**
         * Creates a plain object from this DeploymentStatus message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeploymentStatus.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DeploymentStatus to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DeploymentStatus.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DeploymentStatus;
    })();

    bentoml.ApplyDeploymentRequest = (function() {

        /**
         * Properties of an ApplyDeploymentRequest.
         * @typedef bentoml.ApplyDeploymentRequest$Properties
         * @type {Object}
         * @property {bentoml.Deployment$Properties} [deployment] ApplyDeploymentRequest deployment.
         */

        /**
         * Constructs a new ApplyDeploymentRequest.
         * @exports bentoml.ApplyDeploymentRequest
         * @constructor
         * @param {bentoml.ApplyDeploymentRequest$Properties=} [properties] Properties to set
         */
        function ApplyDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ApplyDeploymentRequest deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        ApplyDeploymentRequest.prototype.deployment = null;

        /**
         * Creates a new ApplyDeploymentRequest instance using the specified properties.
         * @param {bentoml.ApplyDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest instance
         */
        ApplyDeploymentRequest.create = function create(properties) {
            return new ApplyDeploymentRequest(properties);
        };

        /**
         * Encodes the specified ApplyDeploymentRequest message. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentRequest$Properties} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment && message.hasOwnProperty("deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ApplyDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentRequest$Properties} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ApplyDeploymentRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.deployment = $root.bentoml.Deployment.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an ApplyDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ApplyDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment != null) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         */
        ApplyDeploymentRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ApplyDeploymentRequest)
                return object;
            let message = new $root.bentoml.ApplyDeploymentRequest();
            if (object.deployment != null) {
                if (typeof object.deployment !== "object")
                    throw TypeError(".bentoml.ApplyDeploymentRequest.deployment: object expected");
                message.deployment = $root.bentoml.Deployment.fromObject(object.deployment);
            }
            return message;
        };

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ApplyDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         */
        ApplyDeploymentRequest.from = ApplyDeploymentRequest.fromObject;

        /**
         * Creates a plain object from an ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.ApplyDeploymentRequest} message ApplyDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ApplyDeploymentRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.deployment = null;
            if (message.deployment != null && message.hasOwnProperty("deployment"))
                object.deployment = $root.bentoml.Deployment.toObject(message.deployment, options);
            return object;
        };

        /**
         * Creates a plain object from this ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ApplyDeploymentRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ApplyDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ApplyDeploymentRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return ApplyDeploymentRequest;
    })();

    bentoml.ApplyDeploymentResponse = (function() {

        /**
         * Properties of an ApplyDeploymentResponse.
         * @typedef bentoml.ApplyDeploymentResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] ApplyDeploymentResponse status.
         * @property {bentoml.Deployment$Properties} [deployment] ApplyDeploymentResponse deployment.
         */

        /**
         * Constructs a new ApplyDeploymentResponse.
         * @exports bentoml.ApplyDeploymentResponse
         * @constructor
         * @param {bentoml.ApplyDeploymentResponse$Properties=} [properties] Properties to set
         */
        function ApplyDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ApplyDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        ApplyDeploymentResponse.prototype.status = null;

        /**
         * ApplyDeploymentResponse deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        ApplyDeploymentResponse.prototype.deployment = null;

        /**
         * Creates a new ApplyDeploymentResponse instance using the specified properties.
         * @param {bentoml.ApplyDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse instance
         */
        ApplyDeploymentResponse.create = function create(properties) {
            return new ApplyDeploymentResponse(properties);
        };

        /**
         * Encodes the specified ApplyDeploymentResponse message. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentResponse$Properties} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployment && message.hasOwnProperty("deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ApplyDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @param {bentoml.ApplyDeploymentResponse$Properties} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ApplyDeploymentResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.deployment = $root.bentoml.Deployment.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an ApplyDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ApplyDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployment != null) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         */
        ApplyDeploymentResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ApplyDeploymentResponse)
                return object;
            let message = new $root.bentoml.ApplyDeploymentResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.ApplyDeploymentResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.deployment != null) {
                if (typeof object.deployment !== "object")
                    throw TypeError(".bentoml.ApplyDeploymentResponse.deployment: object expected");
                message.deployment = $root.bentoml.Deployment.fromObject(object.deployment);
            }
            return message;
        };

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ApplyDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         */
        ApplyDeploymentResponse.from = ApplyDeploymentResponse.fromObject;

        /**
         * Creates a plain object from an ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.ApplyDeploymentResponse} message ApplyDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ApplyDeploymentResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.deployment = null;
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.deployment != null && message.hasOwnProperty("deployment"))
                object.deployment = $root.bentoml.Deployment.toObject(message.deployment, options);
            return object;
        };

        /**
         * Creates a plain object from this ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ApplyDeploymentResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ApplyDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ApplyDeploymentResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return ApplyDeploymentResponse;
    })();

    bentoml.DeleteDeploymentRequest = (function() {

        /**
         * Properties of a DeleteDeploymentRequest.
         * @typedef bentoml.DeleteDeploymentRequest$Properties
         * @type {Object}
         * @property {string} [deployment_name] DeleteDeploymentRequest deployment_name.
         * @property {string} [namespace] DeleteDeploymentRequest namespace.
         * @property {boolean} [force_delete] DeleteDeploymentRequest force_delete.
         */

        /**
         * Constructs a new DeleteDeploymentRequest.
         * @exports bentoml.DeleteDeploymentRequest
         * @constructor
         * @param {bentoml.DeleteDeploymentRequest$Properties=} [properties] Properties to set
         */
        function DeleteDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeleteDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        DeleteDeploymentRequest.prototype.deployment_name = "";

        /**
         * DeleteDeploymentRequest namespace.
         * @type {string|undefined}
         */
        DeleteDeploymentRequest.prototype.namespace = "";

        /**
         * DeleteDeploymentRequest force_delete.
         * @type {boolean|undefined}
         */
        DeleteDeploymentRequest.prototype.force_delete = false;

        /**
         * Creates a new DeleteDeploymentRequest instance using the specified properties.
         * @param {bentoml.DeleteDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest instance
         */
        DeleteDeploymentRequest.create = function create(properties) {
            return new DeleteDeploymentRequest(properties);
        };

        /**
         * Encodes the specified DeleteDeploymentRequest message. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentRequest$Properties} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            if (message.force_delete != null && message.hasOwnProperty("force_delete"))
                writer.uint32(/* id 3, wireType 0 =*/24).bool(message.force_delete);
            return writer;
        };

        /**
         * Encodes the specified DeleteDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentRequest$Properties} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeleteDeploymentRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.deployment_name = reader.string();
                    break;
                case 2:
                    message.namespace = reader.string();
                    break;
                case 3:
                    message.force_delete = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeleteDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DeleteDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null)
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null)
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.force_delete != null)
                if (typeof message.force_delete !== "boolean")
                    return "force_delete: boolean expected";
            return null;
        };

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         */
        DeleteDeploymentRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DeleteDeploymentRequest)
                return object;
            let message = new $root.bentoml.DeleteDeploymentRequest();
            if (object.deployment_name != null)
                message.deployment_name = String(object.deployment_name);
            if (object.namespace != null)
                message.namespace = String(object.namespace);
            if (object.force_delete != null)
                message.force_delete = Boolean(object.force_delete);
            return message;
        };

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeleteDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         */
        DeleteDeploymentRequest.from = DeleteDeploymentRequest.fromObject;

        /**
         * Creates a plain object from a DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.DeleteDeploymentRequest} message DeleteDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeleteDeploymentRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.deployment_name = "";
                object.namespace = "";
                object.force_delete = false;
            }
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                object.deployment_name = message.deployment_name;
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                object.namespace = message.namespace;
            if (message.force_delete != null && message.hasOwnProperty("force_delete"))
                object.force_delete = message.force_delete;
            return object;
        };

        /**
         * Creates a plain object from this DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeleteDeploymentRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DeleteDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DeleteDeploymentRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DeleteDeploymentRequest;
    })();

    bentoml.DeleteDeploymentResponse = (function() {

        /**
         * Properties of a DeleteDeploymentResponse.
         * @typedef bentoml.DeleteDeploymentResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] DeleteDeploymentResponse status.
         */

        /**
         * Constructs a new DeleteDeploymentResponse.
         * @exports bentoml.DeleteDeploymentResponse
         * @constructor
         * @param {bentoml.DeleteDeploymentResponse$Properties=} [properties] Properties to set
         */
        function DeleteDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeleteDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        DeleteDeploymentResponse.prototype.status = null;

        /**
         * Creates a new DeleteDeploymentResponse instance using the specified properties.
         * @param {bentoml.DeleteDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse instance
         */
        DeleteDeploymentResponse.create = function create(properties) {
            return new DeleteDeploymentResponse(properties);
        };

        /**
         * Encodes the specified DeleteDeploymentResponse message. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentResponse$Properties} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeleteDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DeleteDeploymentResponse$Properties} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DeleteDeploymentResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeleteDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DeleteDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         */
        DeleteDeploymentResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DeleteDeploymentResponse)
                return object;
            let message = new $root.bentoml.DeleteDeploymentResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.DeleteDeploymentResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            return message;
        };

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DeleteDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         */
        DeleteDeploymentResponse.from = DeleteDeploymentResponse.fromObject;

        /**
         * Creates a plain object from a DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.DeleteDeploymentResponse} message DeleteDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeleteDeploymentResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            return object;
        };

        /**
         * Creates a plain object from this DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DeleteDeploymentResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DeleteDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DeleteDeploymentResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DeleteDeploymentResponse;
    })();

    bentoml.GetDeploymentRequest = (function() {

        /**
         * Properties of a GetDeploymentRequest.
         * @typedef bentoml.GetDeploymentRequest$Properties
         * @type {Object}
         * @property {string} [deployment_name] GetDeploymentRequest deployment_name.
         * @property {string} [namespace] GetDeploymentRequest namespace.
         */

        /**
         * Constructs a new GetDeploymentRequest.
         * @exports bentoml.GetDeploymentRequest
         * @constructor
         * @param {bentoml.GetDeploymentRequest$Properties=} [properties] Properties to set
         */
        function GetDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        GetDeploymentRequest.prototype.deployment_name = "";

        /**
         * GetDeploymentRequest namespace.
         * @type {string|undefined}
         */
        GetDeploymentRequest.prototype.namespace = "";

        /**
         * Creates a new GetDeploymentRequest instance using the specified properties.
         * @param {bentoml.GetDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest instance
         */
        GetDeploymentRequest.create = function create(properties) {
            return new GetDeploymentRequest(properties);
        };

        /**
         * Encodes the specified GetDeploymentRequest message. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param {bentoml.GetDeploymentRequest$Properties} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            return writer;
        };

        /**
         * Encodes the specified GetDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @param {bentoml.GetDeploymentRequest$Properties} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.GetDeploymentRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.deployment_name = reader.string();
                    break;
                case 2:
                    message.namespace = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        GetDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null)
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null)
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            return null;
        };

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         */
        GetDeploymentRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.GetDeploymentRequest)
                return object;
            let message = new $root.bentoml.GetDeploymentRequest();
            if (object.deployment_name != null)
                message.deployment_name = String(object.deployment_name);
            if (object.namespace != null)
                message.namespace = String(object.namespace);
            return message;
        };

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         */
        GetDeploymentRequest.from = GetDeploymentRequest.fromObject;

        /**
         * Creates a plain object from a GetDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.GetDeploymentRequest} message GetDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetDeploymentRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.deployment_name = "";
                object.namespace = "";
            }
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                object.deployment_name = message.deployment_name;
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                object.namespace = message.namespace;
            return object;
        };

        /**
         * Creates a plain object from this GetDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetDeploymentRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this GetDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        GetDeploymentRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GetDeploymentRequest;
    })();

    bentoml.GetDeploymentResponse = (function() {

        /**
         * Properties of a GetDeploymentResponse.
         * @typedef bentoml.GetDeploymentResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] GetDeploymentResponse status.
         * @property {bentoml.Deployment$Properties} [deployment] GetDeploymentResponse deployment.
         */

        /**
         * Constructs a new GetDeploymentResponse.
         * @exports bentoml.GetDeploymentResponse
         * @constructor
         * @param {bentoml.GetDeploymentResponse$Properties=} [properties] Properties to set
         */
        function GetDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        GetDeploymentResponse.prototype.status = null;

        /**
         * GetDeploymentResponse deployment.
         * @type {bentoml.Deployment$Properties|undefined}
         */
        GetDeploymentResponse.prototype.deployment = null;

        /**
         * Creates a new GetDeploymentResponse instance using the specified properties.
         * @param {bentoml.GetDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse instance
         */
        GetDeploymentResponse.create = function create(properties) {
            return new GetDeploymentResponse(properties);
        };

        /**
         * Encodes the specified GetDeploymentResponse message. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param {bentoml.GetDeploymentResponse$Properties} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployment && message.hasOwnProperty("deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified GetDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @param {bentoml.GetDeploymentResponse$Properties} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.GetDeploymentResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.deployment = $root.bentoml.Deployment.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        GetDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployment != null) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         */
        GetDeploymentResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.GetDeploymentResponse)
                return object;
            let message = new $root.bentoml.GetDeploymentResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.GetDeploymentResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.deployment != null) {
                if (typeof object.deployment !== "object")
                    throw TypeError(".bentoml.GetDeploymentResponse.deployment: object expected");
                message.deployment = $root.bentoml.Deployment.fromObject(object.deployment);
            }
            return message;
        };

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         */
        GetDeploymentResponse.from = GetDeploymentResponse.fromObject;

        /**
         * Creates a plain object from a GetDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetDeploymentResponse} message GetDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetDeploymentResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.deployment = null;
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.deployment != null && message.hasOwnProperty("deployment"))
                object.deployment = $root.bentoml.Deployment.toObject(message.deployment, options);
            return object;
        };

        /**
         * Creates a plain object from this GetDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetDeploymentResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this GetDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        GetDeploymentResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GetDeploymentResponse;
    })();

    bentoml.DescribeDeploymentRequest = (function() {

        /**
         * Properties of a DescribeDeploymentRequest.
         * @typedef bentoml.DescribeDeploymentRequest$Properties
         * @type {Object}
         * @property {string} [deployment_name] DescribeDeploymentRequest deployment_name.
         * @property {string} [namespace] DescribeDeploymentRequest namespace.
         */

        /**
         * Constructs a new DescribeDeploymentRequest.
         * @exports bentoml.DescribeDeploymentRequest
         * @constructor
         * @param {bentoml.DescribeDeploymentRequest$Properties=} [properties] Properties to set
         */
        function DescribeDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DescribeDeploymentRequest deployment_name.
         * @type {string|undefined}
         */
        DescribeDeploymentRequest.prototype.deployment_name = "";

        /**
         * DescribeDeploymentRequest namespace.
         * @type {string|undefined}
         */
        DescribeDeploymentRequest.prototype.namespace = "";

        /**
         * Creates a new DescribeDeploymentRequest instance using the specified properties.
         * @param {bentoml.DescribeDeploymentRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest instance
         */
        DescribeDeploymentRequest.create = function create(properties) {
            return new DescribeDeploymentRequest(properties);
        };

        /**
         * Encodes the specified DescribeDeploymentRequest message. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentRequest$Properties} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            return writer;
        };

        /**
         * Encodes the specified DescribeDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentRequest$Properties} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DescribeDeploymentRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.deployment_name = reader.string();
                    break;
                case 2:
                    message.namespace = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DescribeDeploymentRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DescribeDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null)
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null)
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            return null;
        };

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         */
        DescribeDeploymentRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DescribeDeploymentRequest)
                return object;
            let message = new $root.bentoml.DescribeDeploymentRequest();
            if (object.deployment_name != null)
                message.deployment_name = String(object.deployment_name);
            if (object.namespace != null)
                message.namespace = String(object.namespace);
            return message;
        };

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DescribeDeploymentRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         */
        DescribeDeploymentRequest.from = DescribeDeploymentRequest.fromObject;

        /**
         * Creates a plain object from a DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @param {bentoml.DescribeDeploymentRequest} message DescribeDeploymentRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DescribeDeploymentRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.deployment_name = "";
                object.namespace = "";
            }
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                object.deployment_name = message.deployment_name;
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                object.namespace = message.namespace;
            return object;
        };

        /**
         * Creates a plain object from this DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DescribeDeploymentRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DescribeDeploymentRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DescribeDeploymentRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DescribeDeploymentRequest;
    })();

    bentoml.DescribeDeploymentResponse = (function() {

        /**
         * Properties of a DescribeDeploymentResponse.
         * @typedef bentoml.DescribeDeploymentResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] DescribeDeploymentResponse status.
         * @property {bentoml.DeploymentState$Properties} [state] DescribeDeploymentResponse state.
         */

        /**
         * Constructs a new DescribeDeploymentResponse.
         * @exports bentoml.DescribeDeploymentResponse
         * @constructor
         * @param {bentoml.DescribeDeploymentResponse$Properties=} [properties] Properties to set
         */
        function DescribeDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DescribeDeploymentResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        DescribeDeploymentResponse.prototype.status = null;

        /**
         * DescribeDeploymentResponse state.
         * @type {bentoml.DeploymentState$Properties|undefined}
         */
        DescribeDeploymentResponse.prototype.state = null;

        /**
         * Creates a new DescribeDeploymentResponse instance using the specified properties.
         * @param {bentoml.DescribeDeploymentResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse instance
         */
        DescribeDeploymentResponse.create = function create(properties) {
            return new DescribeDeploymentResponse(properties);
        };

        /**
         * Encodes the specified DescribeDeploymentResponse message. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentResponse$Properties} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.state && message.hasOwnProperty("state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DescribeDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @param {bentoml.DescribeDeploymentResponse$Properties} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DescribeDeploymentResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.state = $root.bentoml.DeploymentState.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DescribeDeploymentResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DescribeDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.state != null) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            return null;
        };

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         */
        DescribeDeploymentResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DescribeDeploymentResponse)
                return object;
            let message = new $root.bentoml.DescribeDeploymentResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.DescribeDeploymentResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.state != null) {
                if (typeof object.state !== "object")
                    throw TypeError(".bentoml.DescribeDeploymentResponse.state: object expected");
                message.state = $root.bentoml.DeploymentState.fromObject(object.state);
            }
            return message;
        };

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DescribeDeploymentResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         */
        DescribeDeploymentResponse.from = DescribeDeploymentResponse.fromObject;

        /**
         * Creates a plain object from a DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @param {bentoml.DescribeDeploymentResponse} message DescribeDeploymentResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DescribeDeploymentResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.state = null;
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.state != null && message.hasOwnProperty("state"))
                object.state = $root.bentoml.DeploymentState.toObject(message.state, options);
            return object;
        };

        /**
         * Creates a plain object from this DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DescribeDeploymentResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DescribeDeploymentResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DescribeDeploymentResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DescribeDeploymentResponse;
    })();

    bentoml.ListDeploymentsRequest = (function() {

        /**
         * Properties of a ListDeploymentsRequest.
         * @typedef bentoml.ListDeploymentsRequest$Properties
         * @type {Object}
         * @property {string} [namespace] ListDeploymentsRequest namespace.
         * @property {number} [offset] ListDeploymentsRequest offset.
         * @property {number} [limit] ListDeploymentsRequest limit.
         * @property {bentoml.DeploymentSpec.DeploymentOperator} [operator] ListDeploymentsRequest operator.
         * @property {bentoml.ListDeploymentsRequest.SORTABLE_COLUMN} [order_by] ListDeploymentsRequest order_by.
         * @property {boolean} [ascending_order] ListDeploymentsRequest ascending_order.
         * @property {string} [labels_query] ListDeploymentsRequest labels_query.
         */

        /**
         * Constructs a new ListDeploymentsRequest.
         * @exports bentoml.ListDeploymentsRequest
         * @constructor
         * @param {bentoml.ListDeploymentsRequest$Properties=} [properties] Properties to set
         */
        function ListDeploymentsRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListDeploymentsRequest namespace.
         * @type {string|undefined}
         */
        ListDeploymentsRequest.prototype.namespace = "";

        /**
         * ListDeploymentsRequest offset.
         * @type {number|undefined}
         */
        ListDeploymentsRequest.prototype.offset = 0;

        /**
         * ListDeploymentsRequest limit.
         * @type {number|undefined}
         */
        ListDeploymentsRequest.prototype.limit = 0;

        /**
         * ListDeploymentsRequest operator.
         * @type {bentoml.DeploymentSpec.DeploymentOperator|undefined}
         */
        ListDeploymentsRequest.prototype.operator = 0;

        /**
         * ListDeploymentsRequest order_by.
         * @type {bentoml.ListDeploymentsRequest.SORTABLE_COLUMN|undefined}
         */
        ListDeploymentsRequest.prototype.order_by = 0;

        /**
         * ListDeploymentsRequest ascending_order.
         * @type {boolean|undefined}
         */
        ListDeploymentsRequest.prototype.ascending_order = false;

        /**
         * ListDeploymentsRequest labels_query.
         * @type {string|undefined}
         */
        ListDeploymentsRequest.prototype.labels_query = "";

        /**
         * Creates a new ListDeploymentsRequest instance using the specified properties.
         * @param {bentoml.ListDeploymentsRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest instance
         */
        ListDeploymentsRequest.create = function create(properties) {
            return new ListDeploymentsRequest(properties);
        };

        /**
         * Encodes the specified ListDeploymentsRequest message. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param {bentoml.ListDeploymentsRequest$Properties} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.namespace);
            if (message.offset != null && message.hasOwnProperty("offset"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.offset);
            if (message.limit != null && message.hasOwnProperty("limit"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.limit);
            if (message.operator != null && message.hasOwnProperty("operator"))
                writer.uint32(/* id 4, wireType 0 =*/32).uint32(message.operator);
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                writer.uint32(/* id 5, wireType 0 =*/40).uint32(message.order_by);
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                writer.uint32(/* id 6, wireType 0 =*/48).bool(message.ascending_order);
            if (message.labels_query != null && message.hasOwnProperty("labels_query"))
                writer.uint32(/* id 7, wireType 2 =*/58).string(message.labels_query);
            return writer;
        };

        /**
         * Encodes the specified ListDeploymentsRequest message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @param {bentoml.ListDeploymentsRequest$Properties} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ListDeploymentsRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.namespace = reader.string();
                    break;
                case 2:
                    message.offset = reader.int32();
                    break;
                case 3:
                    message.limit = reader.int32();
                    break;
                case 4:
                    message.operator = reader.uint32();
                    break;
                case 5:
                    message.order_by = reader.uint32();
                    break;
                case 6:
                    message.ascending_order = reader.bool();
                    break;
                case 7:
                    message.labels_query = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListDeploymentsRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ListDeploymentsRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.namespace != null)
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.offset != null)
                if (!$util.isInteger(message.offset))
                    return "offset: integer expected";
            if (message.limit != null)
                if (!$util.isInteger(message.limit))
                    return "limit: integer expected";
            if (message.operator != null)
                switch (message.operator) {
                default:
                    return "operator: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                    break;
                }
            if (message.order_by != null)
                switch (message.order_by) {
                default:
                    return "order_by: enum value expected";
                case 0:
                case 1:
                    break;
                }
            if (message.ascending_order != null)
                if (typeof message.ascending_order !== "boolean")
                    return "ascending_order: boolean expected";
            if (message.labels_query != null)
                if (!$util.isString(message.labels_query))
                    return "labels_query: string expected";
            return null;
        };

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         */
        ListDeploymentsRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ListDeploymentsRequest)
                return object;
            let message = new $root.bentoml.ListDeploymentsRequest();
            if (object.namespace != null)
                message.namespace = String(object.namespace);
            if (object.offset != null)
                message.offset = object.offset | 0;
            if (object.limit != null)
                message.limit = object.limit | 0;
            switch (object.operator) {
            case "UNSET":
            case 0:
                message.operator = 0;
                break;
            case "CUSTOM":
            case 1:
                message.operator = 1;
                break;
            case "AWS_SAGEMAKER":
            case 2:
                message.operator = 2;
                break;
            case "AWS_LAMBDA":
            case 3:
                message.operator = 3;
                break;
            case "AZURE_FUNCTIONS":
            case 4:
                message.operator = 4;
                break;
            }
            switch (object.order_by) {
            case "created_at":
            case 0:
                message.order_by = 0;
                break;
            case "name":
            case 1:
                message.order_by = 1;
                break;
            }
            if (object.ascending_order != null)
                message.ascending_order = Boolean(object.ascending_order);
            if (object.labels_query != null)
                message.labels_query = String(object.labels_query);
            return message;
        };

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListDeploymentsRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         */
        ListDeploymentsRequest.from = ListDeploymentsRequest.fromObject;

        /**
         * Creates a plain object from a ListDeploymentsRequest message. Also converts values to other types if specified.
         * @param {bentoml.ListDeploymentsRequest} message ListDeploymentsRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListDeploymentsRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.namespace = "";
                object.offset = 0;
                object.limit = 0;
                object.operator = options.enums === String ? "UNSET" : 0;
                object.order_by = options.enums === String ? "created_at" : 0;
                object.ascending_order = false;
                object.labels_query = "";
            }
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                object.namespace = message.namespace;
            if (message.offset != null && message.hasOwnProperty("offset"))
                object.offset = message.offset;
            if (message.limit != null && message.hasOwnProperty("limit"))
                object.limit = message.limit;
            if (message.operator != null && message.hasOwnProperty("operator"))
                object.operator = options.enums === String ? $root.bentoml.DeploymentSpec.DeploymentOperator[message.operator] : message.operator;
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                object.order_by = options.enums === String ? $root.bentoml.ListDeploymentsRequest.SORTABLE_COLUMN[message.order_by] : message.order_by;
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                object.ascending_order = message.ascending_order;
            if (message.labels_query != null && message.hasOwnProperty("labels_query"))
                object.labels_query = message.labels_query;
            return object;
        };

        /**
         * Creates a plain object from this ListDeploymentsRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListDeploymentsRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ListDeploymentsRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ListDeploymentsRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * SORTABLE_COLUMN enum.
         * @name SORTABLE_COLUMN
         * @memberof bentoml.ListDeploymentsRequest
         * @enum {number}
         * @property {number} created_at=0 created_at value
         * @property {number} name=1 name value
         */
        ListDeploymentsRequest.SORTABLE_COLUMN = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "created_at"] = 0;
            values[valuesById[1] = "name"] = 1;
            return values;
        })();

        return ListDeploymentsRequest;
    })();

    bentoml.ListDeploymentsResponse = (function() {

        /**
         * Properties of a ListDeploymentsResponse.
         * @typedef bentoml.ListDeploymentsResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] ListDeploymentsResponse status.
         * @property {Array.<bentoml.Deployment$Properties>} [deployments] ListDeploymentsResponse deployments.
         */

        /**
         * Constructs a new ListDeploymentsResponse.
         * @exports bentoml.ListDeploymentsResponse
         * @constructor
         * @param {bentoml.ListDeploymentsResponse$Properties=} [properties] Properties to set
         */
        function ListDeploymentsResponse(properties) {
            this.deployments = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListDeploymentsResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        ListDeploymentsResponse.prototype.status = null;

        /**
         * ListDeploymentsResponse deployments.
         * @type {Array.<bentoml.Deployment$Properties>|undefined}
         */
        ListDeploymentsResponse.prototype.deployments = $util.emptyArray;

        /**
         * Creates a new ListDeploymentsResponse instance using the specified properties.
         * @param {bentoml.ListDeploymentsResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse instance
         */
        ListDeploymentsResponse.create = function create(properties) {
            return new ListDeploymentsResponse(properties);
        };

        /**
         * Encodes the specified ListDeploymentsResponse message. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param {bentoml.ListDeploymentsResponse$Properties} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployments && message.deployments.length)
                for (let i = 0; i < message.deployments.length; ++i)
                    $root.bentoml.Deployment.encode(message.deployments[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ListDeploymentsResponse message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @param {bentoml.ListDeploymentsResponse$Properties} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ListDeploymentsResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    if (!(message.deployments && message.deployments.length))
                        message.deployments = [];
                    message.deployments.push($root.bentoml.Deployment.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListDeploymentsResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ListDeploymentsResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployments != null) {
                if (!Array.isArray(message.deployments))
                    return "deployments: array expected";
                for (let i = 0; i < message.deployments.length; ++i) {
                    let error = $root.bentoml.Deployment.verify(message.deployments[i]);
                    if (error)
                        return "deployments." + error;
                }
            }
            return null;
        };

        /**
         * Creates a ListDeploymentsResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         */
        ListDeploymentsResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ListDeploymentsResponse)
                return object;
            let message = new $root.bentoml.ListDeploymentsResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.ListDeploymentsResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.deployments) {
                if (!Array.isArray(object.deployments))
                    throw TypeError(".bentoml.ListDeploymentsResponse.deployments: array expected");
                message.deployments = [];
                for (let i = 0; i < object.deployments.length; ++i) {
                    if (typeof object.deployments[i] !== "object")
                        throw TypeError(".bentoml.ListDeploymentsResponse.deployments: object expected");
                    message.deployments[i] = $root.bentoml.Deployment.fromObject(object.deployments[i]);
                }
            }
            return message;
        };

        /**
         * Creates a ListDeploymentsResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListDeploymentsResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         */
        ListDeploymentsResponse.from = ListDeploymentsResponse.fromObject;

        /**
         * Creates a plain object from a ListDeploymentsResponse message. Also converts values to other types if specified.
         * @param {bentoml.ListDeploymentsResponse} message ListDeploymentsResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListDeploymentsResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.arrays || options.defaults)
                object.deployments = [];
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.deployments && message.deployments.length) {
                object.deployments = [];
                for (let j = 0; j < message.deployments.length; ++j)
                    object.deployments[j] = $root.bentoml.Deployment.toObject(message.deployments[j], options);
            }
            return object;
        };

        /**
         * Creates a plain object from this ListDeploymentsResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListDeploymentsResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ListDeploymentsResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ListDeploymentsResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return ListDeploymentsResponse;
    })();

    bentoml.Status = (function() {

        /**
         * Properties of a Status.
         * @typedef bentoml.Status$Properties
         * @type {Object}
         * @property {bentoml.Status.Code} [status_code] Status status_code.
         * @property {string} [error_message] Status error_message.
         */

        /**
         * Constructs a new Status.
         * @exports bentoml.Status
         * @constructor
         * @param {bentoml.Status$Properties=} [properties] Properties to set
         */
        function Status(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * Status status_code.
         * @type {bentoml.Status.Code|undefined}
         */
        Status.prototype.status_code = 0;

        /**
         * Status error_message.
         * @type {string|undefined}
         */
        Status.prototype.error_message = "";

        /**
         * Creates a new Status instance using the specified properties.
         * @param {bentoml.Status$Properties=} [properties] Properties to set
         * @returns {bentoml.Status} Status instance
         */
        Status.create = function create(properties) {
            return new Status(properties);
        };

        /**
         * Encodes the specified Status message. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param {bentoml.Status$Properties} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Status.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status_code != null && message.hasOwnProperty("status_code"))
                writer.uint32(/* id 1, wireType 0 =*/8).uint32(message.status_code);
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.error_message);
            return writer;
        };

        /**
         * Encodes the specified Status message, length delimited. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @param {bentoml.Status$Properties} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Status.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Status message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Status} Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Status.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.Status();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status_code = reader.uint32();
                    break;
                case 2:
                    message.error_message = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Status message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Status} Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Status.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Status message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        Status.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status_code != null)
                switch (message.status_code) {
                default:
                    return "status_code: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 16:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 20:
                    break;
                }
            if (message.error_message != null)
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            return null;
        };

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Status} Status
         */
        Status.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.Status)
                return object;
            let message = new $root.bentoml.Status();
            switch (object.status_code) {
            case "OK":
            case 0:
                message.status_code = 0;
                break;
            case "CANCELLED":
            case 1:
                message.status_code = 1;
                break;
            case "UNKNOWN":
            case 2:
                message.status_code = 2;
                break;
            case "INVALID_ARGUMENT":
            case 3:
                message.status_code = 3;
                break;
            case "DEADLINE_EXCEEDED":
            case 4:
                message.status_code = 4;
                break;
            case "NOT_FOUND":
            case 5:
                message.status_code = 5;
                break;
            case "ALREADY_EXISTS":
            case 6:
                message.status_code = 6;
                break;
            case "PERMISSION_DENIED":
            case 7:
                message.status_code = 7;
                break;
            case "UNAUTHENTICATED":
            case 16:
                message.status_code = 16;
                break;
            case "RESOURCE_EXHAUSTED":
            case 8:
                message.status_code = 8;
                break;
            case "FAILED_PRECONDITION":
            case 9:
                message.status_code = 9;
                break;
            case "ABORTED":
            case 10:
                message.status_code = 10;
                break;
            case "OUT_OF_RANGE":
            case 11:
                message.status_code = 11;
                break;
            case "UNIMPLEMENTED":
            case 12:
                message.status_code = 12;
                break;
            case "INTERNAL":
            case 13:
                message.status_code = 13;
                break;
            case "UNAVAILABLE":
            case 14:
                message.status_code = 14;
                break;
            case "DATA_LOSS":
            case 15:
                message.status_code = 15;
                break;
            case "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_":
            case 20:
                message.status_code = 20;
                break;
            }
            if (object.error_message != null)
                message.error_message = String(object.error_message);
            return message;
        };

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Status.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Status} Status
         */
        Status.from = Status.fromObject;

        /**
         * Creates a plain object from a Status message. Also converts values to other types if specified.
         * @param {bentoml.Status} message Status
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Status.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status_code = options.enums === String ? "OK" : 0;
                object.error_message = "";
            }
            if (message.status_code != null && message.hasOwnProperty("status_code"))
                object.status_code = options.enums === String ? $root.bentoml.Status.Code[message.status_code] : message.status_code;
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                object.error_message = message.error_message;
            return object;
        };

        /**
         * Creates a plain object from this Status message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Status.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this Status to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        Status.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

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
        Status.Code = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "OK"] = 0;
            values[valuesById[1] = "CANCELLED"] = 1;
            values[valuesById[2] = "UNKNOWN"] = 2;
            values[valuesById[3] = "INVALID_ARGUMENT"] = 3;
            values[valuesById[4] = "DEADLINE_EXCEEDED"] = 4;
            values[valuesById[5] = "NOT_FOUND"] = 5;
            values[valuesById[6] = "ALREADY_EXISTS"] = 6;
            values[valuesById[7] = "PERMISSION_DENIED"] = 7;
            values[valuesById[16] = "UNAUTHENTICATED"] = 16;
            values[valuesById[8] = "RESOURCE_EXHAUSTED"] = 8;
            values[valuesById[9] = "FAILED_PRECONDITION"] = 9;
            values[valuesById[10] = "ABORTED"] = 10;
            values[valuesById[11] = "OUT_OF_RANGE"] = 11;
            values[valuesById[12] = "UNIMPLEMENTED"] = 12;
            values[valuesById[13] = "INTERNAL"] = 13;
            values[valuesById[14] = "UNAVAILABLE"] = 14;
            values[valuesById[15] = "DATA_LOSS"] = 15;
            values[valuesById[20] = "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_"] = 20;
            return values;
        })();

        return Status;
    })();

    bentoml.BentoUri = (function() {

        /**
         * Properties of a BentoUri.
         * @typedef bentoml.BentoUri$Properties
         * @type {Object}
         * @property {bentoml.BentoUri.StorageType} [type] BentoUri type.
         * @property {string} [uri] BentoUri uri.
         * @property {string} [cloud_presigned_url] BentoUri cloud_presigned_url.
         */

        /**
         * Constructs a new BentoUri.
         * @exports bentoml.BentoUri
         * @constructor
         * @param {bentoml.BentoUri$Properties=} [properties] Properties to set
         */
        function BentoUri(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * BentoUri type.
         * @type {bentoml.BentoUri.StorageType|undefined}
         */
        BentoUri.prototype.type = 0;

        /**
         * BentoUri uri.
         * @type {string|undefined}
         */
        BentoUri.prototype.uri = "";

        /**
         * BentoUri cloud_presigned_url.
         * @type {string|undefined}
         */
        BentoUri.prototype.cloud_presigned_url = "";

        /**
         * Creates a new BentoUri instance using the specified properties.
         * @param {bentoml.BentoUri$Properties=} [properties] Properties to set
         * @returns {bentoml.BentoUri} BentoUri instance
         */
        BentoUri.create = function create(properties) {
            return new BentoUri(properties);
        };

        /**
         * Encodes the specified BentoUri message. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param {bentoml.BentoUri$Properties} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoUri.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.type != null && message.hasOwnProperty("type"))
                writer.uint32(/* id 1, wireType 0 =*/8).uint32(message.type);
            if (message.uri != null && message.hasOwnProperty("uri"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.uri);
            if (message.cloud_presigned_url != null && message.hasOwnProperty("cloud_presigned_url"))
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.cloud_presigned_url);
            return writer;
        };

        /**
         * Encodes the specified BentoUri message, length delimited. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @param {bentoml.BentoUri$Properties} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoUri.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a BentoUri message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.BentoUri} BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoUri.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.BentoUri();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.type = reader.uint32();
                    break;
                case 2:
                    message.uri = reader.string();
                    break;
                case 3:
                    message.cloud_presigned_url = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a BentoUri message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoUri} BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoUri.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a BentoUri message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        BentoUri.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.type != null)
                switch (message.type) {
                default:
                    return "type: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                    break;
                }
            if (message.uri != null)
                if (!$util.isString(message.uri))
                    return "uri: string expected";
            if (message.cloud_presigned_url != null)
                if (!$util.isString(message.cloud_presigned_url))
                    return "cloud_presigned_url: string expected";
            return null;
        };

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoUri} BentoUri
         */
        BentoUri.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.BentoUri)
                return object;
            let message = new $root.bentoml.BentoUri();
            switch (object.type) {
            case "UNSET":
            case 0:
                message.type = 0;
                break;
            case "LOCAL":
            case 1:
                message.type = 1;
                break;
            case "S3":
            case 2:
                message.type = 2;
                break;
            case "GCS":
            case 3:
                message.type = 3;
                break;
            case "AZURE_BLOB_STORAGE":
            case 4:
                message.type = 4;
                break;
            case "HDFS":
            case 5:
                message.type = 5;
                break;
            }
            if (object.uri != null)
                message.uri = String(object.uri);
            if (object.cloud_presigned_url != null)
                message.cloud_presigned_url = String(object.cloud_presigned_url);
            return message;
        };

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.BentoUri.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoUri} BentoUri
         */
        BentoUri.from = BentoUri.fromObject;

        /**
         * Creates a plain object from a BentoUri message. Also converts values to other types if specified.
         * @param {bentoml.BentoUri} message BentoUri
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        BentoUri.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.type = options.enums === String ? "UNSET" : 0;
                object.uri = "";
                object.cloud_presigned_url = "";
            }
            if (message.type != null && message.hasOwnProperty("type"))
                object.type = options.enums === String ? $root.bentoml.BentoUri.StorageType[message.type] : message.type;
            if (message.uri != null && message.hasOwnProperty("uri"))
                object.uri = message.uri;
            if (message.cloud_presigned_url != null && message.hasOwnProperty("cloud_presigned_url"))
                object.cloud_presigned_url = message.cloud_presigned_url;
            return object;
        };

        /**
         * Creates a plain object from this BentoUri message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        BentoUri.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this BentoUri to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        BentoUri.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

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
        BentoUri.StorageType = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "UNSET"] = 0;
            values[valuesById[1] = "LOCAL"] = 1;
            values[valuesById[2] = "S3"] = 2;
            values[valuesById[3] = "GCS"] = 3;
            values[valuesById[4] = "AZURE_BLOB_STORAGE"] = 4;
            values[valuesById[5] = "HDFS"] = 5;
            return values;
        })();

        return BentoUri;
    })();

    bentoml.BentoServiceMetadata = (function() {

        /**
         * Properties of a BentoServiceMetadata.
         * @typedef bentoml.BentoServiceMetadata$Properties
         * @type {Object}
         * @property {string} [name] BentoServiceMetadata name.
         * @property {string} [version] BentoServiceMetadata version.
         * @property {google.protobuf.Timestamp$Properties} [created_at] BentoServiceMetadata created_at.
         * @property {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties} [env] BentoServiceMetadata env.
         * @property {Array.<bentoml.BentoServiceMetadata.BentoArtifact$Properties>} [artifacts] BentoServiceMetadata artifacts.
         * @property {Array.<bentoml.BentoServiceMetadata.BentoServiceApi$Properties>} [apis] BentoServiceMetadata apis.
         */

        /**
         * Constructs a new BentoServiceMetadata.
         * @exports bentoml.BentoServiceMetadata
         * @constructor
         * @param {bentoml.BentoServiceMetadata$Properties=} [properties] Properties to set
         */
        function BentoServiceMetadata(properties) {
            this.artifacts = [];
            this.apis = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * BentoServiceMetadata name.
         * @type {string|undefined}
         */
        BentoServiceMetadata.prototype.name = "";

        /**
         * BentoServiceMetadata version.
         * @type {string|undefined}
         */
        BentoServiceMetadata.prototype.version = "";

        /**
         * BentoServiceMetadata created_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        BentoServiceMetadata.prototype.created_at = null;

        /**
         * BentoServiceMetadata env.
         * @type {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties|undefined}
         */
        BentoServiceMetadata.prototype.env = null;

        /**
         * BentoServiceMetadata artifacts.
         * @type {Array.<bentoml.BentoServiceMetadata.BentoArtifact$Properties>|undefined}
         */
        BentoServiceMetadata.prototype.artifacts = $util.emptyArray;

        /**
         * BentoServiceMetadata apis.
         * @type {Array.<bentoml.BentoServiceMetadata.BentoServiceApi$Properties>|undefined}
         */
        BentoServiceMetadata.prototype.apis = $util.emptyArray;

        /**
         * Creates a new BentoServiceMetadata instance using the specified properties.
         * @param {bentoml.BentoServiceMetadata$Properties=} [properties] Properties to set
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata instance
         */
        BentoServiceMetadata.create = function create(properties) {
            return new BentoServiceMetadata(properties);
        };

        /**
         * Encodes the specified BentoServiceMetadata message. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param {bentoml.BentoServiceMetadata$Properties} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoServiceMetadata.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.name != null && message.hasOwnProperty("name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
            if (message.version != null && message.hasOwnProperty("version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            if (message.created_at && message.hasOwnProperty("created_at"))
                $root.google.protobuf.Timestamp.encode(message.created_at, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.env && message.hasOwnProperty("env"))
                $root.bentoml.BentoServiceMetadata.BentoServiceEnv.encode(message.env, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.artifacts && message.artifacts.length)
                for (let i = 0; i < message.artifacts.length; ++i)
                    $root.bentoml.BentoServiceMetadata.BentoArtifact.encode(message.artifacts[i], writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            if (message.apis && message.apis.length)
                for (let i = 0; i < message.apis.length; ++i)
                    $root.bentoml.BentoServiceMetadata.BentoServiceApi.encode(message.apis[i], writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified BentoServiceMetadata message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @param {bentoml.BentoServiceMetadata$Properties} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoServiceMetadata.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoServiceMetadata.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.BentoServiceMetadata();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.version = reader.string();
                    break;
                case 3:
                    message.created_at = $root.google.protobuf.Timestamp.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.env = $root.bentoml.BentoServiceMetadata.BentoServiceEnv.decode(reader, reader.uint32());
                    break;
                case 5:
                    if (!(message.artifacts && message.artifacts.length))
                        message.artifacts = [];
                    message.artifacts.push($root.bentoml.BentoServiceMetadata.BentoArtifact.decode(reader, reader.uint32()));
                    break;
                case 6:
                    if (!(message.apis && message.apis.length))
                        message.apis = [];
                    message.apis.push($root.bentoml.BentoServiceMetadata.BentoServiceApi.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoServiceMetadata.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a BentoServiceMetadata message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        BentoServiceMetadata.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.name != null)
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.version != null)
                if (!$util.isString(message.version))
                    return "version: string expected";
            if (message.created_at != null) {
                let error = $root.google.protobuf.Timestamp.verify(message.created_at);
                if (error)
                    return "created_at." + error;
            }
            if (message.env != null) {
                let error = $root.bentoml.BentoServiceMetadata.BentoServiceEnv.verify(message.env);
                if (error)
                    return "env." + error;
            }
            if (message.artifacts != null) {
                if (!Array.isArray(message.artifacts))
                    return "artifacts: array expected";
                for (let i = 0; i < message.artifacts.length; ++i) {
                    let error = $root.bentoml.BentoServiceMetadata.BentoArtifact.verify(message.artifacts[i]);
                    if (error)
                        return "artifacts." + error;
                }
            }
            if (message.apis != null) {
                if (!Array.isArray(message.apis))
                    return "apis: array expected";
                for (let i = 0; i < message.apis.length; ++i) {
                    let error = $root.bentoml.BentoServiceMetadata.BentoServiceApi.verify(message.apis[i]);
                    if (error)
                        return "apis." + error;
                }
            }
            return null;
        };

        /**
         * Creates a BentoServiceMetadata message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         */
        BentoServiceMetadata.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.BentoServiceMetadata)
                return object;
            let message = new $root.bentoml.BentoServiceMetadata();
            if (object.name != null)
                message.name = String(object.name);
            if (object.version != null)
                message.version = String(object.version);
            if (object.created_at != null) {
                if (typeof object.created_at !== "object")
                    throw TypeError(".bentoml.BentoServiceMetadata.created_at: object expected");
                message.created_at = $root.google.protobuf.Timestamp.fromObject(object.created_at);
            }
            if (object.env != null) {
                if (typeof object.env !== "object")
                    throw TypeError(".bentoml.BentoServiceMetadata.env: object expected");
                message.env = $root.bentoml.BentoServiceMetadata.BentoServiceEnv.fromObject(object.env);
            }
            if (object.artifacts) {
                if (!Array.isArray(object.artifacts))
                    throw TypeError(".bentoml.BentoServiceMetadata.artifacts: array expected");
                message.artifacts = [];
                for (let i = 0; i < object.artifacts.length; ++i) {
                    if (typeof object.artifacts[i] !== "object")
                        throw TypeError(".bentoml.BentoServiceMetadata.artifacts: object expected");
                    message.artifacts[i] = $root.bentoml.BentoServiceMetadata.BentoArtifact.fromObject(object.artifacts[i]);
                }
            }
            if (object.apis) {
                if (!Array.isArray(object.apis))
                    throw TypeError(".bentoml.BentoServiceMetadata.apis: array expected");
                message.apis = [];
                for (let i = 0; i < object.apis.length; ++i) {
                    if (typeof object.apis[i] !== "object")
                        throw TypeError(".bentoml.BentoServiceMetadata.apis: object expected");
                    message.apis[i] = $root.bentoml.BentoServiceMetadata.BentoServiceApi.fromObject(object.apis[i]);
                }
            }
            return message;
        };

        /**
         * Creates a BentoServiceMetadata message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.BentoServiceMetadata.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         */
        BentoServiceMetadata.from = BentoServiceMetadata.fromObject;

        /**
         * Creates a plain object from a BentoServiceMetadata message. Also converts values to other types if specified.
         * @param {bentoml.BentoServiceMetadata} message BentoServiceMetadata
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        BentoServiceMetadata.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.arrays || options.defaults) {
                object.artifacts = [];
                object.apis = [];
            }
            if (options.defaults) {
                object.name = "";
                object.version = "";
                object.created_at = null;
                object.env = null;
            }
            if (message.name != null && message.hasOwnProperty("name"))
                object.name = message.name;
            if (message.version != null && message.hasOwnProperty("version"))
                object.version = message.version;
            if (message.created_at != null && message.hasOwnProperty("created_at"))
                object.created_at = $root.google.protobuf.Timestamp.toObject(message.created_at, options);
            if (message.env != null && message.hasOwnProperty("env"))
                object.env = $root.bentoml.BentoServiceMetadata.BentoServiceEnv.toObject(message.env, options);
            if (message.artifacts && message.artifacts.length) {
                object.artifacts = [];
                for (let j = 0; j < message.artifacts.length; ++j)
                    object.artifacts[j] = $root.bentoml.BentoServiceMetadata.BentoArtifact.toObject(message.artifacts[j], options);
            }
            if (message.apis && message.apis.length) {
                object.apis = [];
                for (let j = 0; j < message.apis.length; ++j)
                    object.apis[j] = $root.bentoml.BentoServiceMetadata.BentoServiceApi.toObject(message.apis[j], options);
            }
            return object;
        };

        /**
         * Creates a plain object from this BentoServiceMetadata message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        BentoServiceMetadata.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this BentoServiceMetadata to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        BentoServiceMetadata.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        BentoServiceMetadata.BentoServiceEnv = (function() {

            /**
             * Properties of a BentoServiceEnv.
             * @typedef bentoml.BentoServiceMetadata.BentoServiceEnv$Properties
             * @type {Object}
             * @property {string} [setup_sh] BentoServiceEnv setup_sh.
             * @property {string} [conda_env] BentoServiceEnv conda_env.
             * @property {string} [pip_dependencies] BentoServiceEnv pip_dependencies.
             * @property {string} [python_version] BentoServiceEnv python_version.
             * @property {string} [docker_base_image] BentoServiceEnv docker_base_image.
             */

            /**
             * Constructs a new BentoServiceEnv.
             * @exports bentoml.BentoServiceMetadata.BentoServiceEnv
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties=} [properties] Properties to set
             */
            function BentoServiceEnv(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoServiceEnv setup_sh.
             * @type {string|undefined}
             */
            BentoServiceEnv.prototype.setup_sh = "";

            /**
             * BentoServiceEnv conda_env.
             * @type {string|undefined}
             */
            BentoServiceEnv.prototype.conda_env = "";

            /**
             * BentoServiceEnv pip_dependencies.
             * @type {string|undefined}
             */
            BentoServiceEnv.prototype.pip_dependencies = "";

            /**
             * BentoServiceEnv python_version.
             * @type {string|undefined}
             */
            BentoServiceEnv.prototype.python_version = "";

            /**
             * BentoServiceEnv docker_base_image.
             * @type {string|undefined}
             */
            BentoServiceEnv.prototype.docker_base_image = "";

            /**
             * Creates a new BentoServiceEnv instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv instance
             */
            BentoServiceEnv.create = function create(properties) {
                return new BentoServiceEnv(properties);
            };

            /**
             * Encodes the specified BentoServiceEnv message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceEnv.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.setup_sh != null && message.hasOwnProperty("setup_sh"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.setup_sh);
                if (message.conda_env != null && message.hasOwnProperty("conda_env"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.conda_env);
                if (message.pip_dependencies != null && message.hasOwnProperty("pip_dependencies"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.pip_dependencies);
                if (message.python_version != null && message.hasOwnProperty("python_version"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.python_version);
                if (message.docker_base_image != null && message.hasOwnProperty("docker_base_image"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.docker_base_image);
                return writer;
            };

            /**
             * Encodes the specified BentoServiceEnv message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv$Properties} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceEnv.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceEnv.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.BentoServiceMetadata.BentoServiceEnv();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.setup_sh = reader.string();
                        break;
                    case 2:
                        message.conda_env = reader.string();
                        break;
                    case 3:
                        message.pip_dependencies = reader.string();
                        break;
                    case 4:
                        message.python_version = reader.string();
                        break;
                    case 5:
                        message.docker_base_image = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceEnv.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoServiceEnv message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            BentoServiceEnv.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.setup_sh != null)
                    if (!$util.isString(message.setup_sh))
                        return "setup_sh: string expected";
                if (message.conda_env != null)
                    if (!$util.isString(message.conda_env))
                        return "conda_env: string expected";
                if (message.pip_dependencies != null)
                    if (!$util.isString(message.pip_dependencies))
                        return "pip_dependencies: string expected";
                if (message.python_version != null)
                    if (!$util.isString(message.python_version))
                        return "python_version: string expected";
                if (message.docker_base_image != null)
                    if (!$util.isString(message.docker_base_image))
                        return "docker_base_image: string expected";
                return null;
            };

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             */
            BentoServiceEnv.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.BentoServiceMetadata.BentoServiceEnv)
                    return object;
                let message = new $root.bentoml.BentoServiceMetadata.BentoServiceEnv();
                if (object.setup_sh != null)
                    message.setup_sh = String(object.setup_sh);
                if (object.conda_env != null)
                    message.conda_env = String(object.conda_env);
                if (object.pip_dependencies != null)
                    message.pip_dependencies = String(object.pip_dependencies);
                if (object.python_version != null)
                    message.python_version = String(object.python_version);
                if (object.docker_base_image != null)
                    message.docker_base_image = String(object.docker_base_image);
                return message;
            };

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoServiceEnv.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             */
            BentoServiceEnv.from = BentoServiceEnv.fromObject;

            /**
             * Creates a plain object from a BentoServiceEnv message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv} message BentoServiceEnv
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoServiceEnv.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.setup_sh = "";
                    object.conda_env = "";
                    object.pip_dependencies = "";
                    object.python_version = "";
                    object.docker_base_image = "";
                }
                if (message.setup_sh != null && message.hasOwnProperty("setup_sh"))
                    object.setup_sh = message.setup_sh;
                if (message.conda_env != null && message.hasOwnProperty("conda_env"))
                    object.conda_env = message.conda_env;
                if (message.pip_dependencies != null && message.hasOwnProperty("pip_dependencies"))
                    object.pip_dependencies = message.pip_dependencies;
                if (message.python_version != null && message.hasOwnProperty("python_version"))
                    object.python_version = message.python_version;
                if (message.docker_base_image != null && message.hasOwnProperty("docker_base_image"))
                    object.docker_base_image = message.docker_base_image;
                return object;
            };

            /**
             * Creates a plain object from this BentoServiceEnv message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoServiceEnv.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this BentoServiceEnv to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            BentoServiceEnv.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return BentoServiceEnv;
        })();

        BentoServiceMetadata.BentoArtifact = (function() {

            /**
             * Properties of a BentoArtifact.
             * @typedef bentoml.BentoServiceMetadata.BentoArtifact$Properties
             * @type {Object}
             * @property {string} [name] BentoArtifact name.
             * @property {string} [artifact_type] BentoArtifact artifact_type.
             */

            /**
             * Constructs a new BentoArtifact.
             * @exports bentoml.BentoServiceMetadata.BentoArtifact
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties=} [properties] Properties to set
             */
            function BentoArtifact(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoArtifact name.
             * @type {string|undefined}
             */
            BentoArtifact.prototype.name = "";

            /**
             * BentoArtifact artifact_type.
             * @type {string|undefined}
             */
            BentoArtifact.prototype.artifact_type = "";

            /**
             * Creates a new BentoArtifact instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact instance
             */
            BentoArtifact.create = function create(properties) {
                return new BentoArtifact(properties);
            };

            /**
             * Encodes the specified BentoArtifact message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoArtifact.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.artifact_type != null && message.hasOwnProperty("artifact_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.artifact_type);
                return writer;
            };

            /**
             * Encodes the specified BentoArtifact message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact$Properties} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoArtifact.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoArtifact.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.BentoServiceMetadata.BentoArtifact();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.artifact_type = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoArtifact.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoArtifact message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            BentoArtifact.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null)
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.artifact_type != null)
                    if (!$util.isString(message.artifact_type))
                        return "artifact_type: string expected";
                return null;
            };

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             */
            BentoArtifact.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.BentoServiceMetadata.BentoArtifact)
                    return object;
                let message = new $root.bentoml.BentoServiceMetadata.BentoArtifact();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.artifact_type != null)
                    message.artifact_type = String(object.artifact_type);
                return message;
            };

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoArtifact.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             */
            BentoArtifact.from = BentoArtifact.fromObject;

            /**
             * Creates a plain object from a BentoArtifact message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoArtifact} message BentoArtifact
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoArtifact.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.name = "";
                    object.artifact_type = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.artifact_type != null && message.hasOwnProperty("artifact_type"))
                    object.artifact_type = message.artifact_type;
                return object;
            };

            /**
             * Creates a plain object from this BentoArtifact message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoArtifact.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this BentoArtifact to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            BentoArtifact.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return BentoArtifact;
        })();

        BentoServiceMetadata.BentoServiceApi = (function() {

            /**
             * Properties of a BentoServiceApi.
             * @typedef bentoml.BentoServiceMetadata.BentoServiceApi$Properties
             * @type {Object}
             * @property {string} [name] BentoServiceApi name.
             * @property {string} [input_type] BentoServiceApi input_type.
             * @property {string} [docs] BentoServiceApi docs.
             * @property {google.protobuf.Struct$Properties} [input_config] BentoServiceApi input_config.
             * @property {google.protobuf.Struct$Properties} [output_config] BentoServiceApi output_config.
             * @property {string} [output_type] BentoServiceApi output_type.
             * @property {number} [mb_max_latency] BentoServiceApi mb_max_latency.
             * @property {number} [mb_max_batch_size] BentoServiceApi mb_max_batch_size.
             */

            /**
             * Constructs a new BentoServiceApi.
             * @exports bentoml.BentoServiceMetadata.BentoServiceApi
             * @constructor
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties=} [properties] Properties to set
             */
            function BentoServiceApi(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoServiceApi name.
             * @type {string|undefined}
             */
            BentoServiceApi.prototype.name = "";

            /**
             * BentoServiceApi input_type.
             * @type {string|undefined}
             */
            BentoServiceApi.prototype.input_type = "";

            /**
             * BentoServiceApi docs.
             * @type {string|undefined}
             */
            BentoServiceApi.prototype.docs = "";

            /**
             * BentoServiceApi input_config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            BentoServiceApi.prototype.input_config = null;

            /**
             * BentoServiceApi output_config.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            BentoServiceApi.prototype.output_config = null;

            /**
             * BentoServiceApi output_type.
             * @type {string|undefined}
             */
            BentoServiceApi.prototype.output_type = "";

            /**
             * BentoServiceApi mb_max_latency.
             * @type {number|undefined}
             */
            BentoServiceApi.prototype.mb_max_latency = 0;

            /**
             * BentoServiceApi mb_max_batch_size.
             * @type {number|undefined}
             */
            BentoServiceApi.prototype.mb_max_batch_size = 0;

            /**
             * Creates a new BentoServiceApi instance using the specified properties.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi instance
             */
            BentoServiceApi.create = function create(properties) {
                return new BentoServiceApi(properties);
            };

            /**
             * Encodes the specified BentoServiceApi message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceApi.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && message.hasOwnProperty("name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.input_type != null && message.hasOwnProperty("input_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.input_type);
                if (message.docs != null && message.hasOwnProperty("docs"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.docs);
                if (message.input_config && message.hasOwnProperty("input_config"))
                    $root.google.protobuf.Struct.encode(message.input_config, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.output_config && message.hasOwnProperty("output_config"))
                    $root.google.protobuf.Struct.encode(message.output_config, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.output_type != null && message.hasOwnProperty("output_type"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.output_type);
                if (message.mb_max_latency != null && message.hasOwnProperty("mb_max_latency"))
                    writer.uint32(/* id 7, wireType 0 =*/56).int32(message.mb_max_latency);
                if (message.mb_max_batch_size != null && message.hasOwnProperty("mb_max_batch_size"))
                    writer.uint32(/* id 8, wireType 0 =*/64).int32(message.mb_max_batch_size);
                return writer;
            };

            /**
             * Encodes the specified BentoServiceApi message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi$Properties} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceApi.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceApi.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.BentoServiceMetadata.BentoServiceApi();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.input_type = reader.string();
                        break;
                    case 3:
                        message.docs = reader.string();
                        break;
                    case 4:
                        message.input_config = $root.google.protobuf.Struct.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.output_config = $root.google.protobuf.Struct.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.output_type = reader.string();
                        break;
                    case 7:
                        message.mb_max_latency = reader.int32();
                        break;
                    case 8:
                        message.mb_max_batch_size = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceApi.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoServiceApi message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            BentoServiceApi.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null)
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.input_type != null)
                    if (!$util.isString(message.input_type))
                        return "input_type: string expected";
                if (message.docs != null)
                    if (!$util.isString(message.docs))
                        return "docs: string expected";
                if (message.input_config != null) {
                    let error = $root.google.protobuf.Struct.verify(message.input_config);
                    if (error)
                        return "input_config." + error;
                }
                if (message.output_config != null) {
                    let error = $root.google.protobuf.Struct.verify(message.output_config);
                    if (error)
                        return "output_config." + error;
                }
                if (message.output_type != null)
                    if (!$util.isString(message.output_type))
                        return "output_type: string expected";
                if (message.mb_max_latency != null)
                    if (!$util.isInteger(message.mb_max_latency))
                        return "mb_max_latency: integer expected";
                if (message.mb_max_batch_size != null)
                    if (!$util.isInteger(message.mb_max_batch_size))
                        return "mb_max_batch_size: integer expected";
                return null;
            };

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             */
            BentoServiceApi.fromObject = function fromObject(object) {
                if (object instanceof $root.bentoml.BentoServiceMetadata.BentoServiceApi)
                    return object;
                let message = new $root.bentoml.BentoServiceMetadata.BentoServiceApi();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.input_type != null)
                    message.input_type = String(object.input_type);
                if (object.docs != null)
                    message.docs = String(object.docs);
                if (object.input_config != null) {
                    if (typeof object.input_config !== "object")
                        throw TypeError(".bentoml.BentoServiceMetadata.BentoServiceApi.input_config: object expected");
                    message.input_config = $root.google.protobuf.Struct.fromObject(object.input_config);
                }
                if (object.output_config != null) {
                    if (typeof object.output_config !== "object")
                        throw TypeError(".bentoml.BentoServiceMetadata.BentoServiceApi.output_config: object expected");
                    message.output_config = $root.google.protobuf.Struct.fromObject(object.output_config);
                }
                if (object.output_type != null)
                    message.output_type = String(object.output_type);
                if (object.mb_max_latency != null)
                    message.mb_max_latency = object.mb_max_latency | 0;
                if (object.mb_max_batch_size != null)
                    message.mb_max_batch_size = object.mb_max_batch_size | 0;
                return message;
            };

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link bentoml.BentoServiceMetadata.BentoServiceApi.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             */
            BentoServiceApi.from = BentoServiceApi.fromObject;

            /**
             * Creates a plain object from a BentoServiceApi message. Also converts values to other types if specified.
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi} message BentoServiceApi
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoServiceApi.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.name = "";
                    object.input_type = "";
                    object.docs = "";
                    object.input_config = null;
                    object.output_config = null;
                    object.output_type = "";
                    object.mb_max_latency = 0;
                    object.mb_max_batch_size = 0;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.input_type != null && message.hasOwnProperty("input_type"))
                    object.input_type = message.input_type;
                if (message.docs != null && message.hasOwnProperty("docs"))
                    object.docs = message.docs;
                if (message.input_config != null && message.hasOwnProperty("input_config"))
                    object.input_config = $root.google.protobuf.Struct.toObject(message.input_config, options);
                if (message.output_config != null && message.hasOwnProperty("output_config"))
                    object.output_config = $root.google.protobuf.Struct.toObject(message.output_config, options);
                if (message.output_type != null && message.hasOwnProperty("output_type"))
                    object.output_type = message.output_type;
                if (message.mb_max_latency != null && message.hasOwnProperty("mb_max_latency"))
                    object.mb_max_latency = message.mb_max_latency;
                if (message.mb_max_batch_size != null && message.hasOwnProperty("mb_max_batch_size"))
                    object.mb_max_batch_size = message.mb_max_batch_size;
                return object;
            };

            /**
             * Creates a plain object from this BentoServiceApi message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            BentoServiceApi.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this BentoServiceApi to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            BentoServiceApi.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return BentoServiceApi;
        })();

        return BentoServiceMetadata;
    })();

    bentoml.Bento = (function() {

        /**
         * Properties of a Bento.
         * @typedef bentoml.Bento$Properties
         * @type {Object}
         * @property {string} [name] Bento name.
         * @property {string} [version] Bento version.
         * @property {bentoml.BentoUri$Properties} [uri] Bento uri.
         * @property {bentoml.BentoServiceMetadata$Properties} [bento_service_metadata] Bento bento_service_metadata.
         * @property {bentoml.UploadStatus$Properties} [status] Bento status.
         */

        /**
         * Constructs a new Bento.
         * @exports bentoml.Bento
         * @constructor
         * @param {bentoml.Bento$Properties=} [properties] Properties to set
         */
        function Bento(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * Bento name.
         * @type {string|undefined}
         */
        Bento.prototype.name = "";

        /**
         * Bento version.
         * @type {string|undefined}
         */
        Bento.prototype.version = "";

        /**
         * Bento uri.
         * @type {bentoml.BentoUri$Properties|undefined}
         */
        Bento.prototype.uri = null;

        /**
         * Bento bento_service_metadata.
         * @type {bentoml.BentoServiceMetadata$Properties|undefined}
         */
        Bento.prototype.bento_service_metadata = null;

        /**
         * Bento status.
         * @type {bentoml.UploadStatus$Properties|undefined}
         */
        Bento.prototype.status = null;

        /**
         * Creates a new Bento instance using the specified properties.
         * @param {bentoml.Bento$Properties=} [properties] Properties to set
         * @returns {bentoml.Bento} Bento instance
         */
        Bento.create = function create(properties) {
            return new Bento(properties);
        };

        /**
         * Encodes the specified Bento message. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param {bentoml.Bento$Properties} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Bento.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.name != null && message.hasOwnProperty("name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
            if (message.version != null && message.hasOwnProperty("version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            if (message.uri && message.hasOwnProperty("uri"))
                $root.bentoml.BentoUri.encode(message.uri, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.bento_service_metadata && message.hasOwnProperty("bento_service_metadata"))
                $root.bentoml.BentoServiceMetadata.encode(message.bento_service_metadata, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.UploadStatus.encode(message.status, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified Bento message, length delimited. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @param {bentoml.Bento$Properties} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Bento.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Bento message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Bento} Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Bento.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.Bento();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.version = reader.string();
                    break;
                case 3:
                    message.uri = $root.bentoml.BentoUri.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bento_service_metadata = $root.bentoml.BentoServiceMetadata.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.status = $root.bentoml.UploadStatus.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Bento message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Bento} Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Bento.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Bento message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        Bento.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.name != null)
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.version != null)
                if (!$util.isString(message.version))
                    return "version: string expected";
            if (message.uri != null) {
                let error = $root.bentoml.BentoUri.verify(message.uri);
                if (error)
                    return "uri." + error;
            }
            if (message.bento_service_metadata != null) {
                let error = $root.bentoml.BentoServiceMetadata.verify(message.bento_service_metadata);
                if (error)
                    return "bento_service_metadata." + error;
            }
            if (message.status != null) {
                let error = $root.bentoml.UploadStatus.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Bento} Bento
         */
        Bento.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.Bento)
                return object;
            let message = new $root.bentoml.Bento();
            if (object.name != null)
                message.name = String(object.name);
            if (object.version != null)
                message.version = String(object.version);
            if (object.uri != null) {
                if (typeof object.uri !== "object")
                    throw TypeError(".bentoml.Bento.uri: object expected");
                message.uri = $root.bentoml.BentoUri.fromObject(object.uri);
            }
            if (object.bento_service_metadata != null) {
                if (typeof object.bento_service_metadata !== "object")
                    throw TypeError(".bentoml.Bento.bento_service_metadata: object expected");
                message.bento_service_metadata = $root.bentoml.BentoServiceMetadata.fromObject(object.bento_service_metadata);
            }
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.Bento.status: object expected");
                message.status = $root.bentoml.UploadStatus.fromObject(object.status);
            }
            return message;
        };

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Bento.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Bento} Bento
         */
        Bento.from = Bento.fromObject;

        /**
         * Creates a plain object from a Bento message. Also converts values to other types if specified.
         * @param {bentoml.Bento} message Bento
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Bento.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.name = "";
                object.version = "";
                object.uri = null;
                object.bento_service_metadata = null;
                object.status = null;
            }
            if (message.name != null && message.hasOwnProperty("name"))
                object.name = message.name;
            if (message.version != null && message.hasOwnProperty("version"))
                object.version = message.version;
            if (message.uri != null && message.hasOwnProperty("uri"))
                object.uri = $root.bentoml.BentoUri.toObject(message.uri, options);
            if (message.bento_service_metadata != null && message.hasOwnProperty("bento_service_metadata"))
                object.bento_service_metadata = $root.bentoml.BentoServiceMetadata.toObject(message.bento_service_metadata, options);
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.UploadStatus.toObject(message.status, options);
            return object;
        };

        /**
         * Creates a plain object from this Bento message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Bento.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this Bento to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        Bento.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Bento;
    })();

    bentoml.AddBentoRequest = (function() {

        /**
         * Properties of an AddBentoRequest.
         * @typedef bentoml.AddBentoRequest$Properties
         * @type {Object}
         * @property {string} [bento_name] AddBentoRequest bento_name.
         * @property {string} [bento_version] AddBentoRequest bento_version.
         */

        /**
         * Constructs a new AddBentoRequest.
         * @exports bentoml.AddBentoRequest
         * @constructor
         * @param {bentoml.AddBentoRequest$Properties=} [properties] Properties to set
         */
        function AddBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * AddBentoRequest bento_name.
         * @type {string|undefined}
         */
        AddBentoRequest.prototype.bento_name = "";

        /**
         * AddBentoRequest bento_version.
         * @type {string|undefined}
         */
        AddBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new AddBentoRequest instance using the specified properties.
         * @param {bentoml.AddBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.AddBentoRequest} AddBentoRequest instance
         */
        AddBentoRequest.create = function create(properties) {
            return new AddBentoRequest(properties);
        };

        /**
         * Encodes the specified AddBentoRequest message. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param {bentoml.AddBentoRequest$Properties} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified AddBentoRequest message, length delimited. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @param {bentoml.AddBentoRequest$Properties} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.AddBentoRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.bento_version = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an AddBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        AddBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null)
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         */
        AddBentoRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.AddBentoRequest)
                return object;
            let message = new $root.bentoml.AddBentoRequest();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.bento_version != null)
                message.bento_version = String(object.bento_version);
            return message;
        };

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.AddBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         */
        AddBentoRequest.from = AddBentoRequest.fromObject;

        /**
         * Creates a plain object from an AddBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.AddBentoRequest} message AddBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        AddBentoRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.bento_version = "";
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                object.bento_version = message.bento_version;
            return object;
        };

        /**
         * Creates a plain object from this AddBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        AddBentoRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this AddBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        AddBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return AddBentoRequest;
    })();

    bentoml.AddBentoResponse = (function() {

        /**
         * Properties of an AddBentoResponse.
         * @typedef bentoml.AddBentoResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] AddBentoResponse status.
         * @property {bentoml.BentoUri$Properties} [uri] AddBentoResponse uri.
         */

        /**
         * Constructs a new AddBentoResponse.
         * @exports bentoml.AddBentoResponse
         * @constructor
         * @param {bentoml.AddBentoResponse$Properties=} [properties] Properties to set
         */
        function AddBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * AddBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        AddBentoResponse.prototype.status = null;

        /**
         * AddBentoResponse uri.
         * @type {bentoml.BentoUri$Properties|undefined}
         */
        AddBentoResponse.prototype.uri = null;

        /**
         * Creates a new AddBentoResponse instance using the specified properties.
         * @param {bentoml.AddBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.AddBentoResponse} AddBentoResponse instance
         */
        AddBentoResponse.create = function create(properties) {
            return new AddBentoResponse(properties);
        };

        /**
         * Encodes the specified AddBentoResponse message. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param {bentoml.AddBentoResponse$Properties} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.uri && message.hasOwnProperty("uri"))
                $root.bentoml.BentoUri.encode(message.uri, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified AddBentoResponse message, length delimited. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @param {bentoml.AddBentoResponse$Properties} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.AddBentoResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.uri = $root.bentoml.BentoUri.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an AddBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        AddBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.uri != null) {
                let error = $root.bentoml.BentoUri.verify(message.uri);
                if (error)
                    return "uri." + error;
            }
            return null;
        };

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         */
        AddBentoResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.AddBentoResponse)
                return object;
            let message = new $root.bentoml.AddBentoResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.AddBentoResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.uri != null) {
                if (typeof object.uri !== "object")
                    throw TypeError(".bentoml.AddBentoResponse.uri: object expected");
                message.uri = $root.bentoml.BentoUri.fromObject(object.uri);
            }
            return message;
        };

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.AddBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         */
        AddBentoResponse.from = AddBentoResponse.fromObject;

        /**
         * Creates a plain object from an AddBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.AddBentoResponse} message AddBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        AddBentoResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.uri = null;
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.uri != null && message.hasOwnProperty("uri"))
                object.uri = $root.bentoml.BentoUri.toObject(message.uri, options);
            return object;
        };

        /**
         * Creates a plain object from this AddBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        AddBentoResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this AddBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        AddBentoResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return AddBentoResponse;
    })();

    bentoml.UploadStatus = (function() {

        /**
         * Properties of an UploadStatus.
         * @typedef bentoml.UploadStatus$Properties
         * @type {Object}
         * @property {bentoml.UploadStatus.Status} [status] UploadStatus status.
         * @property {google.protobuf.Timestamp$Properties} [updated_at] UploadStatus updated_at.
         * @property {number} [percentage] UploadStatus percentage.
         * @property {string} [error_message] UploadStatus error_message.
         */

        /**
         * Constructs a new UploadStatus.
         * @exports bentoml.UploadStatus
         * @constructor
         * @param {bentoml.UploadStatus$Properties=} [properties] Properties to set
         */
        function UploadStatus(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * UploadStatus status.
         * @type {bentoml.UploadStatus.Status|undefined}
         */
        UploadStatus.prototype.status = 0;

        /**
         * UploadStatus updated_at.
         * @type {google.protobuf.Timestamp$Properties|undefined}
         */
        UploadStatus.prototype.updated_at = null;

        /**
         * UploadStatus percentage.
         * @type {number|undefined}
         */
        UploadStatus.prototype.percentage = 0;

        /**
         * UploadStatus error_message.
         * @type {string|undefined}
         */
        UploadStatus.prototype.error_message = "";

        /**
         * Creates a new UploadStatus instance using the specified properties.
         * @param {bentoml.UploadStatus$Properties=} [properties] Properties to set
         * @returns {bentoml.UploadStatus} UploadStatus instance
         */
        UploadStatus.create = function create(properties) {
            return new UploadStatus(properties);
        };

        /**
         * Encodes the specified UploadStatus message. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param {bentoml.UploadStatus$Properties} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UploadStatus.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && message.hasOwnProperty("status"))
                writer.uint32(/* id 1, wireType 0 =*/8).uint32(message.status);
            if (message.updated_at && message.hasOwnProperty("updated_at"))
                $root.google.protobuf.Timestamp.encode(message.updated_at, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            if (message.percentage != null && message.hasOwnProperty("percentage"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.percentage);
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                writer.uint32(/* id 4, wireType 2 =*/34).string(message.error_message);
            return writer;
        };

        /**
         * Encodes the specified UploadStatus message, length delimited. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @param {bentoml.UploadStatus$Properties} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UploadStatus.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UploadStatus message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UploadStatus} UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UploadStatus.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.UploadStatus();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = reader.uint32();
                    break;
                case 2:
                    message.updated_at = $root.google.protobuf.Timestamp.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.percentage = reader.int32();
                    break;
                case 4:
                    message.error_message = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an UploadStatus message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UploadStatus} UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UploadStatus.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UploadStatus message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        UploadStatus.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null)
                switch (message.status) {
                default:
                    return "status: enum value expected";
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                    break;
                }
            if (message.updated_at != null) {
                let error = $root.google.protobuf.Timestamp.verify(message.updated_at);
                if (error)
                    return "updated_at." + error;
            }
            if (message.percentage != null)
                if (!$util.isInteger(message.percentage))
                    return "percentage: integer expected";
            if (message.error_message != null)
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            return null;
        };

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UploadStatus} UploadStatus
         */
        UploadStatus.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.UploadStatus)
                return object;
            let message = new $root.bentoml.UploadStatus();
            switch (object.status) {
            case "UNINITIALIZED":
            case 0:
                message.status = 0;
                break;
            case "UPLOADING":
            case 1:
                message.status = 1;
                break;
            case "DONE":
            case 2:
                message.status = 2;
                break;
            case "ERROR":
            case 3:
                message.status = 3;
                break;
            case "TIMEOUT":
            case 4:
                message.status = 4;
                break;
            }
            if (object.updated_at != null) {
                if (typeof object.updated_at !== "object")
                    throw TypeError(".bentoml.UploadStatus.updated_at: object expected");
                message.updated_at = $root.google.protobuf.Timestamp.fromObject(object.updated_at);
            }
            if (object.percentage != null)
                message.percentage = object.percentage | 0;
            if (object.error_message != null)
                message.error_message = String(object.error_message);
            return message;
        };

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UploadStatus.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UploadStatus} UploadStatus
         */
        UploadStatus.from = UploadStatus.fromObject;

        /**
         * Creates a plain object from an UploadStatus message. Also converts values to other types if specified.
         * @param {bentoml.UploadStatus} message UploadStatus
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UploadStatus.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = options.enums === String ? "UNINITIALIZED" : 0;
                object.updated_at = null;
                object.percentage = 0;
                object.error_message = "";
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = options.enums === String ? $root.bentoml.UploadStatus.Status[message.status] : message.status;
            if (message.updated_at != null && message.hasOwnProperty("updated_at"))
                object.updated_at = $root.google.protobuf.Timestamp.toObject(message.updated_at, options);
            if (message.percentage != null && message.hasOwnProperty("percentage"))
                object.percentage = message.percentage;
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                object.error_message = message.error_message;
            return object;
        };

        /**
         * Creates a plain object from this UploadStatus message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UploadStatus.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this UploadStatus to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        UploadStatus.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

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
        UploadStatus.Status = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "UNINITIALIZED"] = 0;
            values[valuesById[1] = "UPLOADING"] = 1;
            values[valuesById[2] = "DONE"] = 2;
            values[valuesById[3] = "ERROR"] = 3;
            values[valuesById[4] = "TIMEOUT"] = 4;
            return values;
        })();

        return UploadStatus;
    })();

    bentoml.UpdateBentoRequest = (function() {

        /**
         * Properties of an UpdateBentoRequest.
         * @typedef bentoml.UpdateBentoRequest$Properties
         * @type {Object}
         * @property {string} [bento_name] UpdateBentoRequest bento_name.
         * @property {string} [bento_version] UpdateBentoRequest bento_version.
         * @property {bentoml.UploadStatus$Properties} [upload_status] UpdateBentoRequest upload_status.
         * @property {bentoml.BentoServiceMetadata$Properties} [service_metadata] UpdateBentoRequest service_metadata.
         */

        /**
         * Constructs a new UpdateBentoRequest.
         * @exports bentoml.UpdateBentoRequest
         * @constructor
         * @param {bentoml.UpdateBentoRequest$Properties=} [properties] Properties to set
         */
        function UpdateBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * UpdateBentoRequest bento_name.
         * @type {string|undefined}
         */
        UpdateBentoRequest.prototype.bento_name = "";

        /**
         * UpdateBentoRequest bento_version.
         * @type {string|undefined}
         */
        UpdateBentoRequest.prototype.bento_version = "";

        /**
         * UpdateBentoRequest upload_status.
         * @type {bentoml.UploadStatus$Properties|undefined}
         */
        UpdateBentoRequest.prototype.upload_status = null;

        /**
         * UpdateBentoRequest service_metadata.
         * @type {bentoml.BentoServiceMetadata$Properties|undefined}
         */
        UpdateBentoRequest.prototype.service_metadata = null;

        /**
         * Creates a new UpdateBentoRequest instance using the specified properties.
         * @param {bentoml.UpdateBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest instance
         */
        UpdateBentoRequest.create = function create(properties) {
            return new UpdateBentoRequest(properties);
        };

        /**
         * Encodes the specified UpdateBentoRequest message. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param {bentoml.UpdateBentoRequest$Properties} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            if (message.upload_status && message.hasOwnProperty("upload_status"))
                $root.bentoml.UploadStatus.encode(message.upload_status, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.service_metadata && message.hasOwnProperty("service_metadata"))
                $root.bentoml.BentoServiceMetadata.encode(message.service_metadata, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified UpdateBentoRequest message, length delimited. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @param {bentoml.UpdateBentoRequest$Properties} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.UpdateBentoRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.bento_version = reader.string();
                    break;
                case 3:
                    message.upload_status = $root.bentoml.UploadStatus.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.service_metadata = $root.bentoml.BentoServiceMetadata.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UpdateBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        UpdateBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null)
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            if (message.upload_status != null) {
                let error = $root.bentoml.UploadStatus.verify(message.upload_status);
                if (error)
                    return "upload_status." + error;
            }
            if (message.service_metadata != null) {
                let error = $root.bentoml.BentoServiceMetadata.verify(message.service_metadata);
                if (error)
                    return "service_metadata." + error;
            }
            return null;
        };

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         */
        UpdateBentoRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.UpdateBentoRequest)
                return object;
            let message = new $root.bentoml.UpdateBentoRequest();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.bento_version != null)
                message.bento_version = String(object.bento_version);
            if (object.upload_status != null) {
                if (typeof object.upload_status !== "object")
                    throw TypeError(".bentoml.UpdateBentoRequest.upload_status: object expected");
                message.upload_status = $root.bentoml.UploadStatus.fromObject(object.upload_status);
            }
            if (object.service_metadata != null) {
                if (typeof object.service_metadata !== "object")
                    throw TypeError(".bentoml.UpdateBentoRequest.service_metadata: object expected");
                message.service_metadata = $root.bentoml.BentoServiceMetadata.fromObject(object.service_metadata);
            }
            return message;
        };

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UpdateBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         */
        UpdateBentoRequest.from = UpdateBentoRequest.fromObject;

        /**
         * Creates a plain object from an UpdateBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.UpdateBentoRequest} message UpdateBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UpdateBentoRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.bento_version = "";
                object.upload_status = null;
                object.service_metadata = null;
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                object.bento_version = message.bento_version;
            if (message.upload_status != null && message.hasOwnProperty("upload_status"))
                object.upload_status = $root.bentoml.UploadStatus.toObject(message.upload_status, options);
            if (message.service_metadata != null && message.hasOwnProperty("service_metadata"))
                object.service_metadata = $root.bentoml.BentoServiceMetadata.toObject(message.service_metadata, options);
            return object;
        };

        /**
         * Creates a plain object from this UpdateBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UpdateBentoRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this UpdateBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        UpdateBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return UpdateBentoRequest;
    })();

    bentoml.UpdateBentoResponse = (function() {

        /**
         * Properties of an UpdateBentoResponse.
         * @typedef bentoml.UpdateBentoResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] UpdateBentoResponse status.
         */

        /**
         * Constructs a new UpdateBentoResponse.
         * @exports bentoml.UpdateBentoResponse
         * @constructor
         * @param {bentoml.UpdateBentoResponse$Properties=} [properties] Properties to set
         */
        function UpdateBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * UpdateBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        UpdateBentoResponse.prototype.status = null;

        /**
         * Creates a new UpdateBentoResponse instance using the specified properties.
         * @param {bentoml.UpdateBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse instance
         */
        UpdateBentoResponse.create = function create(properties) {
            return new UpdateBentoResponse(properties);
        };

        /**
         * Encodes the specified UpdateBentoResponse message. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param {bentoml.UpdateBentoResponse$Properties} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified UpdateBentoResponse message, length delimited. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @param {bentoml.UpdateBentoResponse$Properties} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.UpdateBentoResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UpdateBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        UpdateBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         */
        UpdateBentoResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.UpdateBentoResponse)
                return object;
            let message = new $root.bentoml.UpdateBentoResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.UpdateBentoResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            return message;
        };

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.UpdateBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         */
        UpdateBentoResponse.from = UpdateBentoResponse.fromObject;

        /**
         * Creates a plain object from an UpdateBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.UpdateBentoResponse} message UpdateBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UpdateBentoResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            return object;
        };

        /**
         * Creates a plain object from this UpdateBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        UpdateBentoResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this UpdateBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        UpdateBentoResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return UpdateBentoResponse;
    })();

    bentoml.DangerouslyDeleteBentoRequest = (function() {

        /**
         * Properties of a DangerouslyDeleteBentoRequest.
         * @typedef bentoml.DangerouslyDeleteBentoRequest$Properties
         * @type {Object}
         * @property {string} [bento_name] DangerouslyDeleteBentoRequest bento_name.
         * @property {string} [bento_version] DangerouslyDeleteBentoRequest bento_version.
         */

        /**
         * Constructs a new DangerouslyDeleteBentoRequest.
         * @exports bentoml.DangerouslyDeleteBentoRequest
         * @constructor
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties=} [properties] Properties to set
         */
        function DangerouslyDeleteBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DangerouslyDeleteBentoRequest bento_name.
         * @type {string|undefined}
         */
        DangerouslyDeleteBentoRequest.prototype.bento_name = "";

        /**
         * DangerouslyDeleteBentoRequest bento_version.
         * @type {string|undefined}
         */
        DangerouslyDeleteBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new DangerouslyDeleteBentoRequest instance using the specified properties.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest instance
         */
        DangerouslyDeleteBentoRequest.create = function create(properties) {
            return new DangerouslyDeleteBentoRequest(properties);
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoRequest$Properties} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DangerouslyDeleteBentoRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.bento_version = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DangerouslyDeleteBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DangerouslyDeleteBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null)
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         */
        DangerouslyDeleteBentoRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DangerouslyDeleteBentoRequest)
                return object;
            let message = new $root.bentoml.DangerouslyDeleteBentoRequest();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.bento_version != null)
                message.bento_version = String(object.bento_version);
            return message;
        };

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DangerouslyDeleteBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         */
        DangerouslyDeleteBentoRequest.from = DangerouslyDeleteBentoRequest.fromObject;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.DangerouslyDeleteBentoRequest} message DangerouslyDeleteBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DangerouslyDeleteBentoRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.bento_version = "";
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                object.bento_version = message.bento_version;
            return object;
        };

        /**
         * Creates a plain object from this DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DangerouslyDeleteBentoRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DangerouslyDeleteBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DangerouslyDeleteBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DangerouslyDeleteBentoRequest;
    })();

    bentoml.DangerouslyDeleteBentoResponse = (function() {

        /**
         * Properties of a DangerouslyDeleteBentoResponse.
         * @typedef bentoml.DangerouslyDeleteBentoResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] DangerouslyDeleteBentoResponse status.
         */

        /**
         * Constructs a new DangerouslyDeleteBentoResponse.
         * @exports bentoml.DangerouslyDeleteBentoResponse
         * @constructor
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties=} [properties] Properties to set
         */
        function DangerouslyDeleteBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * DangerouslyDeleteBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        DangerouslyDeleteBentoResponse.prototype.status = null;

        /**
         * Creates a new DangerouslyDeleteBentoResponse instance using the specified properties.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse instance
         */
        DangerouslyDeleteBentoResponse.create = function create(properties) {
            return new DangerouslyDeleteBentoResponse(properties);
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @param {bentoml.DangerouslyDeleteBentoResponse$Properties} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.DangerouslyDeleteBentoResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DangerouslyDeleteBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        DangerouslyDeleteBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         */
        DangerouslyDeleteBentoResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.DangerouslyDeleteBentoResponse)
                return object;
            let message = new $root.bentoml.DangerouslyDeleteBentoResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.DangerouslyDeleteBentoResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            return message;
        };

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.DangerouslyDeleteBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         */
        DangerouslyDeleteBentoResponse.from = DangerouslyDeleteBentoResponse.fromObject;

        /**
         * Creates a plain object from a DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.DangerouslyDeleteBentoResponse} message DangerouslyDeleteBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DangerouslyDeleteBentoResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            return object;
        };

        /**
         * Creates a plain object from this DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        DangerouslyDeleteBentoResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this DangerouslyDeleteBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        DangerouslyDeleteBentoResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return DangerouslyDeleteBentoResponse;
    })();

    bentoml.GetBentoRequest = (function() {

        /**
         * Properties of a GetBentoRequest.
         * @typedef bentoml.GetBentoRequest$Properties
         * @type {Object}
         * @property {string} [bento_name] GetBentoRequest bento_name.
         * @property {string} [bento_version] GetBentoRequest bento_version.
         */

        /**
         * Constructs a new GetBentoRequest.
         * @exports bentoml.GetBentoRequest
         * @constructor
         * @param {bentoml.GetBentoRequest$Properties=} [properties] Properties to set
         */
        function GetBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetBentoRequest bento_name.
         * @type {string|undefined}
         */
        GetBentoRequest.prototype.bento_name = "";

        /**
         * GetBentoRequest bento_version.
         * @type {string|undefined}
         */
        GetBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new GetBentoRequest instance using the specified properties.
         * @param {bentoml.GetBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.GetBentoRequest} GetBentoRequest instance
         */
        GetBentoRequest.create = function create(properties) {
            return new GetBentoRequest(properties);
        };

        /**
         * Encodes the specified GetBentoRequest message. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param {bentoml.GetBentoRequest$Properties} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified GetBentoRequest message, length delimited. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @param {bentoml.GetBentoRequest$Properties} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.GetBentoRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.bento_version = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        GetBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null)
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         */
        GetBentoRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.GetBentoRequest)
                return object;
            let message = new $root.bentoml.GetBentoRequest();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.bento_version != null)
                message.bento_version = String(object.bento_version);
            return message;
        };

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         */
        GetBentoRequest.from = GetBentoRequest.fromObject;

        /**
         * Creates a plain object from a GetBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.GetBentoRequest} message GetBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetBentoRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.bento_version = "";
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                object.bento_version = message.bento_version;
            return object;
        };

        /**
         * Creates a plain object from this GetBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetBentoRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this GetBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        GetBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GetBentoRequest;
    })();

    bentoml.GetBentoResponse = (function() {

        /**
         * Properties of a GetBentoResponse.
         * @typedef bentoml.GetBentoResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] GetBentoResponse status.
         * @property {bentoml.Bento$Properties} [bento] GetBentoResponse bento.
         */

        /**
         * Constructs a new GetBentoResponse.
         * @exports bentoml.GetBentoResponse
         * @constructor
         * @param {bentoml.GetBentoResponse$Properties=} [properties] Properties to set
         */
        function GetBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        GetBentoResponse.prototype.status = null;

        /**
         * GetBentoResponse bento.
         * @type {bentoml.Bento$Properties|undefined}
         */
        GetBentoResponse.prototype.bento = null;

        /**
         * Creates a new GetBentoResponse instance using the specified properties.
         * @param {bentoml.GetBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetBentoResponse} GetBentoResponse instance
         */
        GetBentoResponse.create = function create(properties) {
            return new GetBentoResponse(properties);
        };

        /**
         * Encodes the specified GetBentoResponse message. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param {bentoml.GetBentoResponse$Properties} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.bento && message.hasOwnProperty("bento"))
                $root.bentoml.Bento.encode(message.bento, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified GetBentoResponse message, length delimited. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @param {bentoml.GetBentoResponse$Properties} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.GetBentoResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.bento = $root.bentoml.Bento.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        GetBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.bento != null) {
                let error = $root.bentoml.Bento.verify(message.bento);
                if (error)
                    return "bento." + error;
            }
            return null;
        };

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         */
        GetBentoResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.GetBentoResponse)
                return object;
            let message = new $root.bentoml.GetBentoResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.GetBentoResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.bento != null) {
                if (typeof object.bento !== "object")
                    throw TypeError(".bentoml.GetBentoResponse.bento: object expected");
                message.bento = $root.bentoml.Bento.fromObject(object.bento);
            }
            return message;
        };

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         */
        GetBentoResponse.from = GetBentoResponse.fromObject;

        /**
         * Creates a plain object from a GetBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetBentoResponse} message GetBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetBentoResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.bento = null;
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.bento != null && message.hasOwnProperty("bento"))
                object.bento = $root.bentoml.Bento.toObject(message.bento, options);
            return object;
        };

        /**
         * Creates a plain object from this GetBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetBentoResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this GetBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        GetBentoResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GetBentoResponse;
    })();

    bentoml.ListBentoRequest = (function() {

        /**
         * Properties of a ListBentoRequest.
         * @typedef bentoml.ListBentoRequest$Properties
         * @type {Object}
         * @property {string} [bento_name] ListBentoRequest bento_name.
         * @property {number} [offset] ListBentoRequest offset.
         * @property {number} [limit] ListBentoRequest limit.
         * @property {bentoml.ListBentoRequest.SORTABLE_COLUMN} [order_by] ListBentoRequest order_by.
         * @property {boolean} [ascending_order] ListBentoRequest ascending_order.
         */

        /**
         * Constructs a new ListBentoRequest.
         * @exports bentoml.ListBentoRequest
         * @constructor
         * @param {bentoml.ListBentoRequest$Properties=} [properties] Properties to set
         */
        function ListBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListBentoRequest bento_name.
         * @type {string|undefined}
         */
        ListBentoRequest.prototype.bento_name = "";

        /**
         * ListBentoRequest offset.
         * @type {number|undefined}
         */
        ListBentoRequest.prototype.offset = 0;

        /**
         * ListBentoRequest limit.
         * @type {number|undefined}
         */
        ListBentoRequest.prototype.limit = 0;

        /**
         * ListBentoRequest order_by.
         * @type {bentoml.ListBentoRequest.SORTABLE_COLUMN|undefined}
         */
        ListBentoRequest.prototype.order_by = 0;

        /**
         * ListBentoRequest ascending_order.
         * @type {boolean|undefined}
         */
        ListBentoRequest.prototype.ascending_order = false;

        /**
         * Creates a new ListBentoRequest instance using the specified properties.
         * @param {bentoml.ListBentoRequest$Properties=} [properties] Properties to set
         * @returns {bentoml.ListBentoRequest} ListBentoRequest instance
         */
        ListBentoRequest.create = function create(properties) {
            return new ListBentoRequest(properties);
        };

        /**
         * Encodes the specified ListBentoRequest message. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param {bentoml.ListBentoRequest$Properties} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.offset != null && message.hasOwnProperty("offset"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.offset);
            if (message.limit != null && message.hasOwnProperty("limit"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.limit);
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                writer.uint32(/* id 4, wireType 0 =*/32).uint32(message.order_by);
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                writer.uint32(/* id 5, wireType 0 =*/40).bool(message.ascending_order);
            return writer;
        };

        /**
         * Encodes the specified ListBentoRequest message, length delimited. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @param {bentoml.ListBentoRequest$Properties} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoRequest.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ListBentoRequest();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.bento_name = reader.string();
                    break;
                case 2:
                    message.offset = reader.int32();
                    break;
                case 3:
                    message.limit = reader.int32();
                    break;
                case 4:
                    message.order_by = reader.uint32();
                    break;
                case 5:
                    message.ascending_order = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListBentoRequest message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ListBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null)
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.offset != null)
                if (!$util.isInteger(message.offset))
                    return "offset: integer expected";
            if (message.limit != null)
                if (!$util.isInteger(message.limit))
                    return "limit: integer expected";
            if (message.order_by != null)
                switch (message.order_by) {
                default:
                    return "order_by: enum value expected";
                case 0:
                case 1:
                    break;
                }
            if (message.ascending_order != null)
                if (typeof message.ascending_order !== "boolean")
                    return "ascending_order: boolean expected";
            return null;
        };

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         */
        ListBentoRequest.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ListBentoRequest)
                return object;
            let message = new $root.bentoml.ListBentoRequest();
            if (object.bento_name != null)
                message.bento_name = String(object.bento_name);
            if (object.offset != null)
                message.offset = object.offset | 0;
            if (object.limit != null)
                message.limit = object.limit | 0;
            switch (object.order_by) {
            case "created_at":
            case 0:
                message.order_by = 0;
                break;
            case "name":
            case 1:
                message.order_by = 1;
                break;
            }
            if (object.ascending_order != null)
                message.ascending_order = Boolean(object.ascending_order);
            return message;
        };

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListBentoRequest.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         */
        ListBentoRequest.from = ListBentoRequest.fromObject;

        /**
         * Creates a plain object from a ListBentoRequest message. Also converts values to other types if specified.
         * @param {bentoml.ListBentoRequest} message ListBentoRequest
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListBentoRequest.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.bento_name = "";
                object.offset = 0;
                object.limit = 0;
                object.order_by = options.enums === String ? "created_at" : 0;
                object.ascending_order = false;
            }
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                object.bento_name = message.bento_name;
            if (message.offset != null && message.hasOwnProperty("offset"))
                object.offset = message.offset;
            if (message.limit != null && message.hasOwnProperty("limit"))
                object.limit = message.limit;
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                object.order_by = options.enums === String ? $root.bentoml.ListBentoRequest.SORTABLE_COLUMN[message.order_by] : message.order_by;
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                object.ascending_order = message.ascending_order;
            return object;
        };

        /**
         * Creates a plain object from this ListBentoRequest message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListBentoRequest.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ListBentoRequest to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ListBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * SORTABLE_COLUMN enum.
         * @name SORTABLE_COLUMN
         * @memberof bentoml.ListBentoRequest
         * @enum {number}
         * @property {number} created_at=0 created_at value
         * @property {number} name=1 name value
         */
        ListBentoRequest.SORTABLE_COLUMN = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "created_at"] = 0;
            values[valuesById[1] = "name"] = 1;
            return values;
        })();

        return ListBentoRequest;
    })();

    bentoml.ListBentoResponse = (function() {

        /**
         * Properties of a ListBentoResponse.
         * @typedef bentoml.ListBentoResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] ListBentoResponse status.
         * @property {Array.<bentoml.Bento$Properties>} [bentos] ListBentoResponse bentos.
         */

        /**
         * Constructs a new ListBentoResponse.
         * @exports bentoml.ListBentoResponse
         * @constructor
         * @param {bentoml.ListBentoResponse$Properties=} [properties] Properties to set
         */
        function ListBentoResponse(properties) {
            this.bentos = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListBentoResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        ListBentoResponse.prototype.status = null;

        /**
         * ListBentoResponse bentos.
         * @type {Array.<bentoml.Bento$Properties>|undefined}
         */
        ListBentoResponse.prototype.bentos = $util.emptyArray;

        /**
         * Creates a new ListBentoResponse instance using the specified properties.
         * @param {bentoml.ListBentoResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.ListBentoResponse} ListBentoResponse instance
         */
        ListBentoResponse.create = function create(properties) {
            return new ListBentoResponse(properties);
        };

        /**
         * Encodes the specified ListBentoResponse message. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param {bentoml.ListBentoResponse$Properties} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.bentos && message.bentos.length)
                for (let i = 0; i < message.bentos.length; ++i)
                    $root.bentoml.Bento.encode(message.bentos[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ListBentoResponse message, length delimited. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @param {bentoml.ListBentoResponse$Properties} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.ListBentoResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    if (!(message.bentos && message.bentos.length))
                        message.bentos = [];
                    message.bentos.push($root.bentoml.Bento.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListBentoResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        ListBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.bentos != null) {
                if (!Array.isArray(message.bentos))
                    return "bentos: array expected";
                for (let i = 0; i < message.bentos.length; ++i) {
                    let error = $root.bentoml.Bento.verify(message.bentos[i]);
                    if (error)
                        return "bentos." + error;
                }
            }
            return null;
        };

        /**
         * Creates a ListBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         */
        ListBentoResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.ListBentoResponse)
                return object;
            let message = new $root.bentoml.ListBentoResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.ListBentoResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.bentos) {
                if (!Array.isArray(object.bentos))
                    throw TypeError(".bentoml.ListBentoResponse.bentos: array expected");
                message.bentos = [];
                for (let i = 0; i < object.bentos.length; ++i) {
                    if (typeof object.bentos[i] !== "object")
                        throw TypeError(".bentoml.ListBentoResponse.bentos: object expected");
                    message.bentos[i] = $root.bentoml.Bento.fromObject(object.bentos[i]);
                }
            }
            return message;
        };

        /**
         * Creates a ListBentoResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.ListBentoResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         */
        ListBentoResponse.from = ListBentoResponse.fromObject;

        /**
         * Creates a plain object from a ListBentoResponse message. Also converts values to other types if specified.
         * @param {bentoml.ListBentoResponse} message ListBentoResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListBentoResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.arrays || options.defaults)
                object.bentos = [];
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.bentos && message.bentos.length) {
                object.bentos = [];
                for (let j = 0; j < message.bentos.length; ++j)
                    object.bentos[j] = $root.bentoml.Bento.toObject(message.bentos[j], options);
            }
            return object;
        };

        /**
         * Creates a plain object from this ListBentoResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        ListBentoResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this ListBentoResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        ListBentoResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return ListBentoResponse;
    })();

    bentoml.Yatai = (function() {

        /**
         * Constructs a new Yatai service.
         * @exports bentoml.Yatai
         * @extends $protobuf.rpc.Service
         * @constructor
         * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
         * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
         * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
         */
        function Yatai(rpcImpl, requestDelimited, responseDelimited) {
            $protobuf.rpc.Service.call(this, rpcImpl, requestDelimited, responseDelimited);
        }

        (Yatai.prototype = Object.create($protobuf.rpc.Service.prototype)).constructor = Yatai;

        /**
         * Creates new Yatai service using the specified rpc implementation.
         * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
         * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
         * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
         * @returns {Yatai} RPC service. Useful where requests and/or responses are streamed.
         */
        Yatai.create = function create(rpcImpl, requestDelimited, responseDelimited) {
            return new this(rpcImpl, requestDelimited, responseDelimited);
        };

        /**
         * Callback as used by {@link Yatai#healthCheck}.
         * @typedef Yatai_healthCheck_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.HealthCheckResponse} [response] HealthCheckResponse
         */

        /**
         * Calls HealthCheck.
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @param {Yatai_healthCheck_Callback} callback Node-style callback called with the error, if any, and HealthCheckResponse
         * @returns {undefined}
         */
        Yatai.prototype.healthCheck = function healthCheck(request, callback) {
            return this.rpcCall(healthCheck, $root.google.protobuf.Empty, $root.bentoml.HealthCheckResponse, request, callback);
        };

        /**
         * Calls HealthCheck.
         * @name Yatai#healthCheck
         * @function
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @returns {Promise<bentoml.HealthCheckResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#getYataiServiceVersion}.
         * @typedef Yatai_getYataiServiceVersion_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.GetYataiServiceVersionResponse} [response] GetYataiServiceVersionResponse
         */

        /**
         * Calls GetYataiServiceVersion.
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @param {Yatai_getYataiServiceVersion_Callback} callback Node-style callback called with the error, if any, and GetYataiServiceVersionResponse
         * @returns {undefined}
         */
        Yatai.prototype.getYataiServiceVersion = function getYataiServiceVersion(request, callback) {
            return this.rpcCall(getYataiServiceVersion, $root.google.protobuf.Empty, $root.bentoml.GetYataiServiceVersionResponse, request, callback);
        };

        /**
         * Calls GetYataiServiceVersion.
         * @name Yatai#getYataiServiceVersion
         * @function
         * @param {google.protobuf.Empty|Object.<string,*>} request Empty message or plain object
         * @returns {Promise<bentoml.GetYataiServiceVersionResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#applyDeployment}.
         * @typedef Yatai_applyDeployment_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.ApplyDeploymentResponse} [response] ApplyDeploymentResponse
         */

        /**
         * Calls ApplyDeployment.
         * @param {bentoml.ApplyDeploymentRequest|Object.<string,*>} request ApplyDeploymentRequest message or plain object
         * @param {Yatai_applyDeployment_Callback} callback Node-style callback called with the error, if any, and ApplyDeploymentResponse
         * @returns {undefined}
         */
        Yatai.prototype.applyDeployment = function applyDeployment(request, callback) {
            return this.rpcCall(applyDeployment, $root.bentoml.ApplyDeploymentRequest, $root.bentoml.ApplyDeploymentResponse, request, callback);
        };

        /**
         * Calls ApplyDeployment.
         * @name Yatai#applyDeployment
         * @function
         * @param {bentoml.ApplyDeploymentRequest|Object.<string,*>} request ApplyDeploymentRequest message or plain object
         * @returns {Promise<bentoml.ApplyDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#deleteDeployment}.
         * @typedef Yatai_deleteDeployment_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.DeleteDeploymentResponse} [response] DeleteDeploymentResponse
         */

        /**
         * Calls DeleteDeployment.
         * @param {bentoml.DeleteDeploymentRequest|Object.<string,*>} request DeleteDeploymentRequest message or plain object
         * @param {Yatai_deleteDeployment_Callback} callback Node-style callback called with the error, if any, and DeleteDeploymentResponse
         * @returns {undefined}
         */
        Yatai.prototype.deleteDeployment = function deleteDeployment(request, callback) {
            return this.rpcCall(deleteDeployment, $root.bentoml.DeleteDeploymentRequest, $root.bentoml.DeleteDeploymentResponse, request, callback);
        };

        /**
         * Calls DeleteDeployment.
         * @name Yatai#deleteDeployment
         * @function
         * @param {bentoml.DeleteDeploymentRequest|Object.<string,*>} request DeleteDeploymentRequest message or plain object
         * @returns {Promise<bentoml.DeleteDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#getDeployment}.
         * @typedef Yatai_getDeployment_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.GetDeploymentResponse} [response] GetDeploymentResponse
         */

        /**
         * Calls GetDeployment.
         * @param {bentoml.GetDeploymentRequest|Object.<string,*>} request GetDeploymentRequest message or plain object
         * @param {Yatai_getDeployment_Callback} callback Node-style callback called with the error, if any, and GetDeploymentResponse
         * @returns {undefined}
         */
        Yatai.prototype.getDeployment = function getDeployment(request, callback) {
            return this.rpcCall(getDeployment, $root.bentoml.GetDeploymentRequest, $root.bentoml.GetDeploymentResponse, request, callback);
        };

        /**
         * Calls GetDeployment.
         * @name Yatai#getDeployment
         * @function
         * @param {bentoml.GetDeploymentRequest|Object.<string,*>} request GetDeploymentRequest message or plain object
         * @returns {Promise<bentoml.GetDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#describeDeployment}.
         * @typedef Yatai_describeDeployment_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.DescribeDeploymentResponse} [response] DescribeDeploymentResponse
         */

        /**
         * Calls DescribeDeployment.
         * @param {bentoml.DescribeDeploymentRequest|Object.<string,*>} request DescribeDeploymentRequest message or plain object
         * @param {Yatai_describeDeployment_Callback} callback Node-style callback called with the error, if any, and DescribeDeploymentResponse
         * @returns {undefined}
         */
        Yatai.prototype.describeDeployment = function describeDeployment(request, callback) {
            return this.rpcCall(describeDeployment, $root.bentoml.DescribeDeploymentRequest, $root.bentoml.DescribeDeploymentResponse, request, callback);
        };

        /**
         * Calls DescribeDeployment.
         * @name Yatai#describeDeployment
         * @function
         * @param {bentoml.DescribeDeploymentRequest|Object.<string,*>} request DescribeDeploymentRequest message or plain object
         * @returns {Promise<bentoml.DescribeDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#listDeployments}.
         * @typedef Yatai_listDeployments_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.ListDeploymentsResponse} [response] ListDeploymentsResponse
         */

        /**
         * Calls ListDeployments.
         * @param {bentoml.ListDeploymentsRequest|Object.<string,*>} request ListDeploymentsRequest message or plain object
         * @param {Yatai_listDeployments_Callback} callback Node-style callback called with the error, if any, and ListDeploymentsResponse
         * @returns {undefined}
         */
        Yatai.prototype.listDeployments = function listDeployments(request, callback) {
            return this.rpcCall(listDeployments, $root.bentoml.ListDeploymentsRequest, $root.bentoml.ListDeploymentsResponse, request, callback);
        };

        /**
         * Calls ListDeployments.
         * @name Yatai#listDeployments
         * @function
         * @param {bentoml.ListDeploymentsRequest|Object.<string,*>} request ListDeploymentsRequest message or plain object
         * @returns {Promise<bentoml.ListDeploymentsResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#addBento}.
         * @typedef Yatai_addBento_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.AddBentoResponse} [response] AddBentoResponse
         */

        /**
         * Calls AddBento.
         * @param {bentoml.AddBentoRequest|Object.<string,*>} request AddBentoRequest message or plain object
         * @param {Yatai_addBento_Callback} callback Node-style callback called with the error, if any, and AddBentoResponse
         * @returns {undefined}
         */
        Yatai.prototype.addBento = function addBento(request, callback) {
            return this.rpcCall(addBento, $root.bentoml.AddBentoRequest, $root.bentoml.AddBentoResponse, request, callback);
        };

        /**
         * Calls AddBento.
         * @name Yatai#addBento
         * @function
         * @param {bentoml.AddBentoRequest|Object.<string,*>} request AddBentoRequest message or plain object
         * @returns {Promise<bentoml.AddBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#updateBento}.
         * @typedef Yatai_updateBento_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.UpdateBentoResponse} [response] UpdateBentoResponse
         */

        /**
         * Calls UpdateBento.
         * @param {bentoml.UpdateBentoRequest|Object.<string,*>} request UpdateBentoRequest message or plain object
         * @param {Yatai_updateBento_Callback} callback Node-style callback called with the error, if any, and UpdateBentoResponse
         * @returns {undefined}
         */
        Yatai.prototype.updateBento = function updateBento(request, callback) {
            return this.rpcCall(updateBento, $root.bentoml.UpdateBentoRequest, $root.bentoml.UpdateBentoResponse, request, callback);
        };

        /**
         * Calls UpdateBento.
         * @name Yatai#updateBento
         * @function
         * @param {bentoml.UpdateBentoRequest|Object.<string,*>} request UpdateBentoRequest message or plain object
         * @returns {Promise<bentoml.UpdateBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#getBento}.
         * @typedef Yatai_getBento_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.GetBentoResponse} [response] GetBentoResponse
         */

        /**
         * Calls GetBento.
         * @param {bentoml.GetBentoRequest|Object.<string,*>} request GetBentoRequest message or plain object
         * @param {Yatai_getBento_Callback} callback Node-style callback called with the error, if any, and GetBentoResponse
         * @returns {undefined}
         */
        Yatai.prototype.getBento = function getBento(request, callback) {
            return this.rpcCall(getBento, $root.bentoml.GetBentoRequest, $root.bentoml.GetBentoResponse, request, callback);
        };

        /**
         * Calls GetBento.
         * @name Yatai#getBento
         * @function
         * @param {bentoml.GetBentoRequest|Object.<string,*>} request GetBentoRequest message or plain object
         * @returns {Promise<bentoml.GetBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#dangerouslyDeleteBento}.
         * @typedef Yatai_dangerouslyDeleteBento_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.DangerouslyDeleteBentoResponse} [response] DangerouslyDeleteBentoResponse
         */

        /**
         * Calls DangerouslyDeleteBento.
         * @param {bentoml.DangerouslyDeleteBentoRequest|Object.<string,*>} request DangerouslyDeleteBentoRequest message or plain object
         * @param {Yatai_dangerouslyDeleteBento_Callback} callback Node-style callback called with the error, if any, and DangerouslyDeleteBentoResponse
         * @returns {undefined}
         */
        Yatai.prototype.dangerouslyDeleteBento = function dangerouslyDeleteBento(request, callback) {
            return this.rpcCall(dangerouslyDeleteBento, $root.bentoml.DangerouslyDeleteBentoRequest, $root.bentoml.DangerouslyDeleteBentoResponse, request, callback);
        };

        /**
         * Calls DangerouslyDeleteBento.
         * @name Yatai#dangerouslyDeleteBento
         * @function
         * @param {bentoml.DangerouslyDeleteBentoRequest|Object.<string,*>} request DangerouslyDeleteBentoRequest message or plain object
         * @returns {Promise<bentoml.DangerouslyDeleteBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link Yatai#listBento}.
         * @typedef Yatai_listBento_Callback
         * @type {function}
         * @param {?Error} error Error, if any
         * @param {bentoml.ListBentoResponse} [response] ListBentoResponse
         */

        /**
         * Calls ListBento.
         * @param {bentoml.ListBentoRequest|Object.<string,*>} request ListBentoRequest message or plain object
         * @param {Yatai_listBento_Callback} callback Node-style callback called with the error, if any, and ListBentoResponse
         * @returns {undefined}
         */
        Yatai.prototype.listBento = function listBento(request, callback) {
            return this.rpcCall(listBento, $root.bentoml.ListBentoRequest, $root.bentoml.ListBentoResponse, request, callback);
        };

        /**
         * Calls ListBento.
         * @name Yatai#listBento
         * @function
         * @param {bentoml.ListBentoRequest|Object.<string,*>} request ListBentoRequest message or plain object
         * @returns {Promise<bentoml.ListBentoResponse>} Promise
         * @variation 2
         */

        return Yatai;
    })();

    bentoml.HealthCheckResponse = (function() {

        /**
         * Properties of a HealthCheckResponse.
         * @typedef bentoml.HealthCheckResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] HealthCheckResponse status.
         */

        /**
         * Constructs a new HealthCheckResponse.
         * @exports bentoml.HealthCheckResponse
         * @constructor
         * @param {bentoml.HealthCheckResponse$Properties=} [properties] Properties to set
         */
        function HealthCheckResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * HealthCheckResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        HealthCheckResponse.prototype.status = null;

        /**
         * Creates a new HealthCheckResponse instance using the specified properties.
         * @param {bentoml.HealthCheckResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse instance
         */
        HealthCheckResponse.create = function create(properties) {
            return new HealthCheckResponse(properties);
        };

        /**
         * Encodes the specified HealthCheckResponse message. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param {bentoml.HealthCheckResponse$Properties} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        HealthCheckResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified HealthCheckResponse message, length delimited. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @param {bentoml.HealthCheckResponse$Properties} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        HealthCheckResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        HealthCheckResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.HealthCheckResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        HealthCheckResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a HealthCheckResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        HealthCheckResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         */
        HealthCheckResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.HealthCheckResponse)
                return object;
            let message = new $root.bentoml.HealthCheckResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.HealthCheckResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            return message;
        };

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.HealthCheckResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         */
        HealthCheckResponse.from = HealthCheckResponse.fromObject;

        /**
         * Creates a plain object from a HealthCheckResponse message. Also converts values to other types if specified.
         * @param {bentoml.HealthCheckResponse} message HealthCheckResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        HealthCheckResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.status = null;
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            return object;
        };

        /**
         * Creates a plain object from this HealthCheckResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        HealthCheckResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this HealthCheckResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        HealthCheckResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return HealthCheckResponse;
    })();

    bentoml.GetYataiServiceVersionResponse = (function() {

        /**
         * Properties of a GetYataiServiceVersionResponse.
         * @typedef bentoml.GetYataiServiceVersionResponse$Properties
         * @type {Object}
         * @property {bentoml.Status$Properties} [status] GetYataiServiceVersionResponse status.
         * @property {string} [version] GetYataiServiceVersionResponse version.
         */

        /**
         * Constructs a new GetYataiServiceVersionResponse.
         * @exports bentoml.GetYataiServiceVersionResponse
         * @constructor
         * @param {bentoml.GetYataiServiceVersionResponse$Properties=} [properties] Properties to set
         */
        function GetYataiServiceVersionResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetYataiServiceVersionResponse status.
         * @type {bentoml.Status$Properties|undefined}
         */
        GetYataiServiceVersionResponse.prototype.status = null;

        /**
         * GetYataiServiceVersionResponse version.
         * @type {string|undefined}
         */
        GetYataiServiceVersionResponse.prototype.version = "";

        /**
         * Creates a new GetYataiServiceVersionResponse instance using the specified properties.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties=} [properties] Properties to set
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse instance
         */
        GetYataiServiceVersionResponse.create = function create(properties) {
            return new GetYataiServiceVersionResponse(properties);
        };

        /**
         * Encodes the specified GetYataiServiceVersionResponse message. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetYataiServiceVersionResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status && message.hasOwnProperty("status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.version != null && message.hasOwnProperty("version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            return writer;
        };

        /**
         * Encodes the specified GetYataiServiceVersionResponse message, length delimited. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @param {bentoml.GetYataiServiceVersionResponse$Properties} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetYataiServiceVersionResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetYataiServiceVersionResponse.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.GetYataiServiceVersionResponse();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.status = $root.bentoml.Status.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.version = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetYataiServiceVersionResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetYataiServiceVersionResponse message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        GetYataiServiceVersionResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.version != null)
                if (!$util.isString(message.version))
                    return "version: string expected";
            return null;
        };

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         */
        GetYataiServiceVersionResponse.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.GetYataiServiceVersionResponse)
                return object;
            let message = new $root.bentoml.GetYataiServiceVersionResponse();
            if (object.status != null) {
                if (typeof object.status !== "object")
                    throw TypeError(".bentoml.GetYataiServiceVersionResponse.status: object expected");
                message.status = $root.bentoml.Status.fromObject(object.status);
            }
            if (object.version != null)
                message.version = String(object.version);
            return message;
        };

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.GetYataiServiceVersionResponse.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         */
        GetYataiServiceVersionResponse.from = GetYataiServiceVersionResponse.fromObject;

        /**
         * Creates a plain object from a GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @param {bentoml.GetYataiServiceVersionResponse} message GetYataiServiceVersionResponse
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetYataiServiceVersionResponse.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.status = null;
                object.version = "";
            }
            if (message.status != null && message.hasOwnProperty("status"))
                object.status = $root.bentoml.Status.toObject(message.status, options);
            if (message.version != null && message.hasOwnProperty("version"))
                object.version = message.version;
            return object;
        };

        /**
         * Creates a plain object from this GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        GetYataiServiceVersionResponse.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this GetYataiServiceVersionResponse to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        GetYataiServiceVersionResponse.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return GetYataiServiceVersionResponse;
    })();

    bentoml.Chunk = (function() {

        /**
         * Properties of a Chunk.
         * @typedef bentoml.Chunk$Properties
         * @type {Object}
         * @property {Uint8Array} [content] Chunk content.
         */

        /**
         * Constructs a new Chunk.
         * @exports bentoml.Chunk
         * @constructor
         * @param {bentoml.Chunk$Properties=} [properties] Properties to set
         */
        function Chunk(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    this[keys[i]] = properties[keys[i]];
        }

        /**
         * Chunk content.
         * @type {Uint8Array|undefined}
         */
        Chunk.prototype.content = $util.newBuffer([]);

        /**
         * Creates a new Chunk instance using the specified properties.
         * @param {bentoml.Chunk$Properties=} [properties] Properties to set
         * @returns {bentoml.Chunk} Chunk instance
         */
        Chunk.create = function create(properties) {
            return new Chunk(properties);
        };

        /**
         * Encodes the specified Chunk message. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param {bentoml.Chunk$Properties} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Chunk.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.content && message.hasOwnProperty("content"))
                writer.uint32(/* id 1, wireType 2 =*/10).bytes(message.content);
            return writer;
        };

        /**
         * Encodes the specified Chunk message, length delimited. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @param {bentoml.Chunk$Properties} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Chunk.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Chunk message from the specified reader or buffer.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {bentoml.Chunk} Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Chunk.decode = function decode(reader, length) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.bentoml.Chunk();
            while (reader.pos < end) {
                let tag = reader.uint32();
                switch (tag >>> 3) {
                case 1:
                    message.content = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a Chunk message from the specified reader or buffer, length delimited.
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Chunk} Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Chunk.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Chunk message.
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {?string} `null` if valid, otherwise the reason why it is not
         */
        Chunk.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.content != null)
                if (!(message.content && typeof message.content.length === "number" || $util.isString(message.content)))
                    return "content: buffer expected";
            return null;
        };

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Chunk} Chunk
         */
        Chunk.fromObject = function fromObject(object) {
            if (object instanceof $root.bentoml.Chunk)
                return object;
            let message = new $root.bentoml.Chunk();
            if (object.content != null)
                if (typeof object.content === "string")
                    $util.base64.decode(object.content, message.content = $util.newBuffer($util.base64.length(object.content)), 0);
                else if (object.content.length)
                    message.content = object.content;
            return message;
        };

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * This is an alias of {@link bentoml.Chunk.fromObject}.
         * @function
         * @param {Object.<string,*>} object Plain object
         * @returns {bentoml.Chunk} Chunk
         */
        Chunk.from = Chunk.fromObject;

        /**
         * Creates a plain object from a Chunk message. Also converts values to other types if specified.
         * @param {bentoml.Chunk} message Chunk
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Chunk.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                object.content = options.bytes === String ? "" : [];
            if (message.content != null && message.hasOwnProperty("content"))
                object.content = options.bytes === String ? $util.base64.encode(message.content, 0, message.content.length) : options.bytes === Array ? Array.prototype.slice.call(message.content) : message.content;
            return object;
        };

        /**
         * Creates a plain object from this Chunk message. Also converts values to other types if specified.
         * @param {$protobuf.ConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Chunk.prototype.toObject = function toObject(options) {
            return this.constructor.toObject(this, options);
        };

        /**
         * Converts this Chunk to JSON.
         * @returns {Object.<string,*>} JSON object
         */
        Chunk.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        return Chunk;
    })();

    return bentoml;
})();

export const google = $root.google = (() => {

    /**
     * Namespace google.
     * @exports google
     * @namespace
     */
    const google = {};

    google.protobuf = (function() {

        /**
         * Namespace protobuf.
         * @exports google.protobuf
         * @namespace
         */
        const protobuf = {};

        protobuf.Struct = (function() {

            /**
             * Properties of a Struct.
             * @typedef google.protobuf.Struct$Properties
             * @type {Object}
             * @property {Object.<string,google.protobuf.Value$Properties>} [fields] Struct fields.
             */

            /**
             * Constructs a new Struct.
             * @exports google.protobuf.Struct
             * @constructor
             * @param {google.protobuf.Struct$Properties=} [properties] Properties to set
             */
            function Struct(properties) {
                this.fields = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * Struct fields.
             * @type {Object.<string,google.protobuf.Value$Properties>|undefined}
             */
            Struct.prototype.fields = $util.emptyObject;

            /**
             * Creates a new Struct instance using the specified properties.
             * @param {google.protobuf.Struct$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Struct} Struct instance
             */
            Struct.create = function create(properties) {
                return new Struct(properties);
            };

            /**
             * Encodes the specified Struct message. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param {google.protobuf.Struct$Properties} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Struct.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.fields && message.hasOwnProperty("fields"))
                    for (let keys = Object.keys(message.fields), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 1, wireType 2 =*/10).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.google.protobuf.Value.encode(message.fields[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                return writer;
            };

            /**
             * Encodes the specified Struct message, length delimited. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @param {google.protobuf.Struct$Properties} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Struct.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Struct message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Struct} Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Struct.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Struct(), key;
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        reader.skip().pos++;
                        if (message.fields === $util.emptyObject)
                            message.fields = {};
                        key = reader.string();
                        reader.pos++;
                        message.fields[key] = $root.google.protobuf.Value.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Struct message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Struct} Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Struct.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Struct message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            Struct.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.fields != null) {
                    if (!$util.isObject(message.fields))
                        return "fields: object expected";
                    let key = Object.keys(message.fields);
                    for (let i = 0; i < key.length; ++i) {
                        let error = $root.google.protobuf.Value.verify(message.fields[key[i]]);
                        if (error)
                            return "fields." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a Struct message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Struct} Struct
             */
            Struct.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.Struct)
                    return object;
                let message = new $root.google.protobuf.Struct();
                if (object.fields) {
                    if (typeof object.fields !== "object")
                        throw TypeError(".google.protobuf.Struct.fields: object expected");
                    message.fields = {};
                    for (let keys = Object.keys(object.fields), i = 0; i < keys.length; ++i) {
                        if (typeof object.fields[keys[i]] !== "object")
                            throw TypeError(".google.protobuf.Struct.fields: object expected");
                        message.fields[keys[i]] = $root.google.protobuf.Value.fromObject(object.fields[keys[i]]);
                    }
                }
                return message;
            };

            /**
             * Creates a Struct message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Struct.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Struct} Struct
             */
            Struct.from = Struct.fromObject;

            /**
             * Creates a plain object from a Struct message. Also converts values to other types if specified.
             * @param {google.protobuf.Struct} message Struct
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Struct.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.objects || options.defaults)
                    object.fields = {};
                let keys2;
                if (message.fields && (keys2 = Object.keys(message.fields)).length) {
                    object.fields = {};
                    for (let j = 0; j < keys2.length; ++j)
                        object.fields[keys2[j]] = $root.google.protobuf.Value.toObject(message.fields[keys2[j]], options);
                }
                return object;
            };

            /**
             * Creates a plain object from this Struct message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Struct.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this Struct to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            Struct.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return Struct;
        })();

        protobuf.Value = (function() {

            /**
             * Properties of a Value.
             * @typedef google.protobuf.Value$Properties
             * @type {Object}
             * @property {google.protobuf.NullValue} [nullValue] Value nullValue.
             * @property {number} [numberValue] Value numberValue.
             * @property {string} [stringValue] Value stringValue.
             * @property {boolean} [boolValue] Value boolValue.
             * @property {google.protobuf.Struct$Properties} [structValue] Value structValue.
             * @property {google.protobuf.ListValue$Properties} [listValue] Value listValue.
             */

            /**
             * Constructs a new Value.
             * @exports google.protobuf.Value
             * @constructor
             * @param {google.protobuf.Value$Properties=} [properties] Properties to set
             */
            function Value(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * Value nullValue.
             * @type {google.protobuf.NullValue|undefined}
             */
            Value.prototype.nullValue = 0;

            /**
             * Value numberValue.
             * @type {number|undefined}
             */
            Value.prototype.numberValue = 0;

            /**
             * Value stringValue.
             * @type {string|undefined}
             */
            Value.prototype.stringValue = "";

            /**
             * Value boolValue.
             * @type {boolean|undefined}
             */
            Value.prototype.boolValue = false;

            /**
             * Value structValue.
             * @type {google.protobuf.Struct$Properties|undefined}
             */
            Value.prototype.structValue = null;

            /**
             * Value listValue.
             * @type {google.protobuf.ListValue$Properties|undefined}
             */
            Value.prototype.listValue = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Value kind.
             * @name google.protobuf.Value#kind
             * @type {string|undefined}
             */
            Object.defineProperty(Value.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["nullValue", "numberValue", "stringValue", "boolValue", "structValue", "listValue"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Value instance using the specified properties.
             * @param {google.protobuf.Value$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Value} Value instance
             */
            Value.create = function create(properties) {
                return new Value(properties);
            };

            /**
             * Encodes the specified Value message. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param {google.protobuf.Value$Properties} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Value.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nullValue != null && message.hasOwnProperty("nullValue"))
                    writer.uint32(/* id 1, wireType 0 =*/8).uint32(message.nullValue);
                if (message.numberValue != null && message.hasOwnProperty("numberValue"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.numberValue);
                if (message.stringValue != null && message.hasOwnProperty("stringValue"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.stringValue);
                if (message.boolValue != null && message.hasOwnProperty("boolValue"))
                    writer.uint32(/* id 4, wireType 0 =*/32).bool(message.boolValue);
                if (message.structValue && message.hasOwnProperty("structValue"))
                    $root.google.protobuf.Struct.encode(message.structValue, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.listValue && message.hasOwnProperty("listValue"))
                    $root.google.protobuf.ListValue.encode(message.listValue, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified Value message, length delimited. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @param {google.protobuf.Value$Properties} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Value.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Value message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Value} Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Value.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Value();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.nullValue = reader.uint32();
                        break;
                    case 2:
                        message.numberValue = reader.double();
                        break;
                    case 3:
                        message.stringValue = reader.string();
                        break;
                    case 4:
                        message.boolValue = reader.bool();
                        break;
                    case 5:
                        message.structValue = $root.google.protobuf.Struct.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.listValue = $root.google.protobuf.ListValue.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Value message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Value} Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Value.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Value message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            Value.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let properties = {};
                if (message.nullValue != null) {
                    properties.kind = 1;
                    switch (message.nullValue) {
                    default:
                        return "nullValue: enum value expected";
                    case 0:
                        break;
                    }
                }
                if (message.numberValue != null) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (typeof message.numberValue !== "number")
                        return "numberValue: number expected";
                }
                if (message.stringValue != null) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (!$util.isString(message.stringValue))
                        return "stringValue: string expected";
                }
                if (message.boolValue != null) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (typeof message.boolValue !== "boolean")
                        return "boolValue: boolean expected";
                }
                if (message.structValue != null) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    let error = $root.google.protobuf.Struct.verify(message.structValue);
                    if (error)
                        return "structValue." + error;
                }
                if (message.listValue != null) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    let error = $root.google.protobuf.ListValue.verify(message.listValue);
                    if (error)
                        return "listValue." + error;
                }
                return null;
            };

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Value} Value
             */
            Value.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.Value)
                    return object;
                let message = new $root.google.protobuf.Value();
                switch (object.nullValue) {
                case "NULL_VALUE":
                case 0:
                    message.nullValue = 0;
                    break;
                }
                if (object.numberValue != null)
                    message.numberValue = Number(object.numberValue);
                if (object.stringValue != null)
                    message.stringValue = String(object.stringValue);
                if (object.boolValue != null)
                    message.boolValue = Boolean(object.boolValue);
                if (object.structValue != null) {
                    if (typeof object.structValue !== "object")
                        throw TypeError(".google.protobuf.Value.structValue: object expected");
                    message.structValue = $root.google.protobuf.Struct.fromObject(object.structValue);
                }
                if (object.listValue != null) {
                    if (typeof object.listValue !== "object")
                        throw TypeError(".google.protobuf.Value.listValue: object expected");
                    message.listValue = $root.google.protobuf.ListValue.fromObject(object.listValue);
                }
                return message;
            };

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Value.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Value} Value
             */
            Value.from = Value.fromObject;

            /**
             * Creates a plain object from a Value message. Also converts values to other types if specified.
             * @param {google.protobuf.Value} message Value
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Value.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (message.nullValue != null && message.hasOwnProperty("nullValue")) {
                    object.nullValue = options.enums === String ? $root.google.protobuf.NullValue[message.nullValue] : message.nullValue;
                    if (options.oneofs)
                        object.kind = "nullValue";
                }
                if (message.numberValue != null && message.hasOwnProperty("numberValue")) {
                    object.numberValue = message.numberValue;
                    if (options.oneofs)
                        object.kind = "numberValue";
                }
                if (message.stringValue != null && message.hasOwnProperty("stringValue")) {
                    object.stringValue = message.stringValue;
                    if (options.oneofs)
                        object.kind = "stringValue";
                }
                if (message.boolValue != null && message.hasOwnProperty("boolValue")) {
                    object.boolValue = message.boolValue;
                    if (options.oneofs)
                        object.kind = "boolValue";
                }
                if (message.structValue != null && message.hasOwnProperty("structValue")) {
                    object.structValue = $root.google.protobuf.Struct.toObject(message.structValue, options);
                    if (options.oneofs)
                        object.kind = "structValue";
                }
                if (message.listValue != null && message.hasOwnProperty("listValue")) {
                    object.listValue = $root.google.protobuf.ListValue.toObject(message.listValue, options);
                    if (options.oneofs)
                        object.kind = "listValue";
                }
                return object;
            };

            /**
             * Creates a plain object from this Value message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Value.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this Value to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            Value.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return Value;
        })();

        /**
         * NullValue enum.
         * @name NullValue
         * @memberof google.protobuf
         * @enum {number}
         * @property {number} NULL_VALUE=0 NULL_VALUE value
         */
        protobuf.NullValue = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "NULL_VALUE"] = 0;
            return values;
        })();

        protobuf.ListValue = (function() {

            /**
             * Properties of a ListValue.
             * @typedef google.protobuf.ListValue$Properties
             * @type {Object}
             * @property {Array.<google.protobuf.Value$Properties>} [values] ListValue values.
             */

            /**
             * Constructs a new ListValue.
             * @exports google.protobuf.ListValue
             * @constructor
             * @param {google.protobuf.ListValue$Properties=} [properties] Properties to set
             */
            function ListValue(properties) {
                this.values = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * ListValue values.
             * @type {Array.<google.protobuf.Value$Properties>|undefined}
             */
            ListValue.prototype.values = $util.emptyArray;

            /**
             * Creates a new ListValue instance using the specified properties.
             * @param {google.protobuf.ListValue$Properties=} [properties] Properties to set
             * @returns {google.protobuf.ListValue} ListValue instance
             */
            ListValue.create = function create(properties) {
                return new ListValue(properties);
            };

            /**
             * Encodes the specified ListValue message. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param {google.protobuf.ListValue$Properties} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ListValue.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.values && message.values.length)
                    for (let i = 0; i < message.values.length; ++i)
                        $root.google.protobuf.Value.encode(message.values[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified ListValue message, length delimited. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @param {google.protobuf.ListValue$Properties} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ListValue.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a ListValue message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.ListValue} ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ListValue.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.ListValue();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.values && message.values.length))
                            message.values = [];
                        message.values.push($root.google.protobuf.Value.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a ListValue message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.ListValue} ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ListValue.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a ListValue message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            ListValue.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.values != null) {
                    if (!Array.isArray(message.values))
                        return "values: array expected";
                    for (let i = 0; i < message.values.length; ++i) {
                        let error = $root.google.protobuf.Value.verify(message.values[i]);
                        if (error)
                            return "values." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.ListValue} ListValue
             */
            ListValue.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.ListValue)
                    return object;
                let message = new $root.google.protobuf.ListValue();
                if (object.values) {
                    if (!Array.isArray(object.values))
                        throw TypeError(".google.protobuf.ListValue.values: array expected");
                    message.values = [];
                    for (let i = 0; i < object.values.length; ++i) {
                        if (typeof object.values[i] !== "object")
                            throw TypeError(".google.protobuf.ListValue.values: object expected");
                        message.values[i] = $root.google.protobuf.Value.fromObject(object.values[i]);
                    }
                }
                return message;
            };

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.ListValue.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.ListValue} ListValue
             */
            ListValue.from = ListValue.fromObject;

            /**
             * Creates a plain object from a ListValue message. Also converts values to other types if specified.
             * @param {google.protobuf.ListValue} message ListValue
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ListValue.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.values = [];
                if (message.values && message.values.length) {
                    object.values = [];
                    for (let j = 0; j < message.values.length; ++j)
                        object.values[j] = $root.google.protobuf.Value.toObject(message.values[j], options);
                }
                return object;
            };

            /**
             * Creates a plain object from this ListValue message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ListValue.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this ListValue to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            ListValue.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return ListValue;
        })();

        protobuf.Timestamp = (function() {

            /**
             * Properties of a Timestamp.
             * @typedef google.protobuf.Timestamp$Properties
             * @type {Object}
             * @property {number|Long} [seconds] Timestamp seconds.
             * @property {number} [nanos] Timestamp nanos.
             */

            /**
             * Constructs a new Timestamp.
             * @exports google.protobuf.Timestamp
             * @constructor
             * @param {google.protobuf.Timestamp$Properties=} [properties] Properties to set
             */
            function Timestamp(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * Timestamp seconds.
             * @type {number|Long|undefined}
             */
            Timestamp.prototype.seconds = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * Timestamp nanos.
             * @type {number|undefined}
             */
            Timestamp.prototype.nanos = 0;

            /**
             * Creates a new Timestamp instance using the specified properties.
             * @param {google.protobuf.Timestamp$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Timestamp} Timestamp instance
             */
            Timestamp.create = function create(properties) {
                return new Timestamp(properties);
            };

            /**
             * Encodes the specified Timestamp message. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param {google.protobuf.Timestamp$Properties} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Timestamp.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.seconds != null && message.hasOwnProperty("seconds"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.seconds);
                if (message.nanos != null && message.hasOwnProperty("nanos"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.nanos);
                return writer;
            };

            /**
             * Encodes the specified Timestamp message, length delimited. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @param {google.protobuf.Timestamp$Properties} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Timestamp.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Timestamp message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Timestamp} Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Timestamp.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Timestamp();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.seconds = reader.int64();
                        break;
                    case 2:
                        message.nanos = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Timestamp message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Timestamp} Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Timestamp.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Timestamp message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            Timestamp.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.seconds != null)
                    if (!$util.isInteger(message.seconds) && !(message.seconds && $util.isInteger(message.seconds.low) && $util.isInteger(message.seconds.high)))
                        return "seconds: integer|Long expected";
                if (message.nanos != null)
                    if (!$util.isInteger(message.nanos))
                        return "nanos: integer expected";
                return null;
            };

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Timestamp} Timestamp
             */
            Timestamp.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.Timestamp)
                    return object;
                let message = new $root.google.protobuf.Timestamp();
                if (object.seconds != null)
                    if ($util.Long)
                        (message.seconds = $util.Long.fromValue(object.seconds)).unsigned = false;
                    else if (typeof object.seconds === "string")
                        message.seconds = parseInt(object.seconds, 10);
                    else if (typeof object.seconds === "number")
                        message.seconds = object.seconds;
                    else if (typeof object.seconds === "object")
                        message.seconds = new $util.LongBits(object.seconds.low >>> 0, object.seconds.high >>> 0).toNumber();
                if (object.nanos != null)
                    message.nanos = object.nanos | 0;
                return message;
            };

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Timestamp.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Timestamp} Timestamp
             */
            Timestamp.from = Timestamp.fromObject;

            /**
             * Creates a plain object from a Timestamp message. Also converts values to other types if specified.
             * @param {google.protobuf.Timestamp} message Timestamp
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Timestamp.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.seconds = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.seconds = options.longs === String ? "0" : 0;
                    object.nanos = 0;
                }
                if (message.seconds != null && message.hasOwnProperty("seconds"))
                    if (typeof message.seconds === "number")
                        object.seconds = options.longs === String ? String(message.seconds) : message.seconds;
                    else
                        object.seconds = options.longs === String ? $util.Long.prototype.toString.call(message.seconds) : options.longs === Number ? new $util.LongBits(message.seconds.low >>> 0, message.seconds.high >>> 0).toNumber() : message.seconds;
                if (message.nanos != null && message.hasOwnProperty("nanos"))
                    object.nanos = message.nanos;
                return object;
            };

            /**
             * Creates a plain object from this Timestamp message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Timestamp.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this Timestamp to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            Timestamp.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return Timestamp;
        })();

        protobuf.Empty = (function() {

            /**
             * Properties of an Empty.
             * @typedef google.protobuf.Empty$Properties
             * @type {Object}
             */

            /**
             * Constructs a new Empty.
             * @exports google.protobuf.Empty
             * @constructor
             * @param {google.protobuf.Empty$Properties=} [properties] Properties to set
             */
            function Empty(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new Empty instance using the specified properties.
             * @param {google.protobuf.Empty$Properties=} [properties] Properties to set
             * @returns {google.protobuf.Empty} Empty instance
             */
            Empty.create = function create(properties) {
                return new Empty(properties);
            };

            /**
             * Encodes the specified Empty message. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param {google.protobuf.Empty$Properties} message Empty message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Empty.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                return writer;
            };

            /**
             * Encodes the specified Empty message, length delimited. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @param {google.protobuf.Empty$Properties} message Empty message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Empty.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an Empty message from the specified reader or buffer.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {google.protobuf.Empty} Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Empty.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Empty();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an Empty message from the specified reader or buffer, length delimited.
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Empty} Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Empty.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an Empty message.
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {?string} `null` if valid, otherwise the reason why it is not
             */
            Empty.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                return null;
            };

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Empty} Empty
             */
            Empty.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.Empty)
                    return object;
                return new $root.google.protobuf.Empty();
            };

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * This is an alias of {@link google.protobuf.Empty.fromObject}.
             * @function
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Empty} Empty
             */
            Empty.from = Empty.fromObject;

            /**
             * Creates a plain object from an Empty message. Also converts values to other types if specified.
             * @param {google.protobuf.Empty} message Empty
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Empty.toObject = function toObject() {
                return {};
            };

            /**
             * Creates a plain object from this Empty message. Also converts values to other types if specified.
             * @param {$protobuf.ConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Empty.prototype.toObject = function toObject(options) {
                return this.constructor.toObject(this, options);
            };

            /**
             * Converts this Empty to JSON.
             * @returns {Object.<string,*>} JSON object
             */
            Empty.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return Empty;
        })();

        return protobuf;
    })();

    return google;
})();

export { $root as default };
