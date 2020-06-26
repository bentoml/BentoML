/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
import * as $protobuf from "protobufjs/minimal";

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
         * @memberof bentoml
         * @interface IDeploymentSpec
         * @property {string|null} [bento_name] DeploymentSpec bento_name
         * @property {string|null} [bento_version] DeploymentSpec bento_version
         * @property {bentoml.DeploymentSpec.DeploymentOperator|null} [operator] DeploymentSpec operator
         * @property {bentoml.DeploymentSpec.ICustomOperatorConfig|null} [custom_operator_config] DeploymentSpec custom_operator_config
         * @property {bentoml.DeploymentSpec.ISageMakerOperatorConfig|null} [sagemaker_operator_config] DeploymentSpec sagemaker_operator_config
         * @property {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig|null} [aws_lambda_operator_config] DeploymentSpec aws_lambda_operator_config
         * @property {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig|null} [azure_functions_operator_config] DeploymentSpec azure_functions_operator_config
         */

        /**
         * Constructs a new DeploymentSpec.
         * @memberof bentoml
         * @classdesc Represents a DeploymentSpec.
         * @implements IDeploymentSpec
         * @constructor
         * @param {bentoml.IDeploymentSpec=} [properties] Properties to set
         */
        function DeploymentSpec(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentSpec bento_name.
         * @member {string} bento_name
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.bento_name = "";

        /**
         * DeploymentSpec bento_version.
         * @member {string} bento_version
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.bento_version = "";

        /**
         * DeploymentSpec operator.
         * @member {bentoml.DeploymentSpec.DeploymentOperator} operator
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.operator = 0;

        /**
         * DeploymentSpec custom_operator_config.
         * @member {bentoml.DeploymentSpec.ICustomOperatorConfig|null|undefined} custom_operator_config
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.custom_operator_config = null;

        /**
         * DeploymentSpec sagemaker_operator_config.
         * @member {bentoml.DeploymentSpec.ISageMakerOperatorConfig|null|undefined} sagemaker_operator_config
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.sagemaker_operator_config = null;

        /**
         * DeploymentSpec aws_lambda_operator_config.
         * @member {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig|null|undefined} aws_lambda_operator_config
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.aws_lambda_operator_config = null;

        /**
         * DeploymentSpec azure_functions_operator_config.
         * @member {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig|null|undefined} azure_functions_operator_config
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        DeploymentSpec.prototype.azure_functions_operator_config = null;

        // OneOf field names bound to virtual getters and setters
        let $oneOfFields;

        /**
         * DeploymentSpec deployment_operator_config.
         * @member {"custom_operator_config"|"sagemaker_operator_config"|"aws_lambda_operator_config"|"azure_functions_operator_config"|undefined} deployment_operator_config
         * @memberof bentoml.DeploymentSpec
         * @instance
         */
        Object.defineProperty(DeploymentSpec.prototype, "deployment_operator_config", {
            get: $util.oneOfGetter($oneOfFields = ["custom_operator_config", "sagemaker_operator_config", "aws_lambda_operator_config", "azure_functions_operator_config"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        /**
         * Creates a new DeploymentSpec instance using the specified properties.
         * @function create
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {bentoml.IDeploymentSpec=} [properties] Properties to set
         * @returns {bentoml.DeploymentSpec} DeploymentSpec instance
         */
        DeploymentSpec.create = function create(properties) {
            return new DeploymentSpec(properties);
        };

        /**
         * Encodes the specified DeploymentSpec message. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {bentoml.IDeploymentSpec} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentSpec.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && Object.hasOwnProperty.call(message, "bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            if (message.operator != null && Object.hasOwnProperty.call(message, "operator"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.operator);
            if (message.custom_operator_config != null && Object.hasOwnProperty.call(message, "custom_operator_config"))
                $root.bentoml.DeploymentSpec.CustomOperatorConfig.encode(message.custom_operator_config, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.sagemaker_operator_config != null && Object.hasOwnProperty.call(message, "sagemaker_operator_config"))
                $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.encode(message.sagemaker_operator_config, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            if (message.aws_lambda_operator_config != null && Object.hasOwnProperty.call(message, "aws_lambda_operator_config"))
                $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.encode(message.aws_lambda_operator_config, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
            if (message.azure_functions_operator_config != null && Object.hasOwnProperty.call(message, "azure_functions_operator_config"))
                $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.encode(message.azure_functions_operator_config, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentSpec message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {bentoml.IDeploymentSpec} message DeploymentSpec message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentSpec.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentSpec message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DeploymentSpec
         * @static
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
                    message.operator = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentSpec} DeploymentSpec
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentSpec.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentSpec message.
         * @function verify
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DeploymentSpec.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            let properties = {};
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            if (message.operator != null && message.hasOwnProperty("operator"))
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
            if (message.custom_operator_config != null && message.hasOwnProperty("custom_operator_config")) {
                properties.deployment_operator_config = 1;
                {
                    let error = $root.bentoml.DeploymentSpec.CustomOperatorConfig.verify(message.custom_operator_config);
                    if (error)
                        return "custom_operator_config." + error;
                }
            }
            if (message.sagemaker_operator_config != null && message.hasOwnProperty("sagemaker_operator_config")) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                {
                    let error = $root.bentoml.DeploymentSpec.SageMakerOperatorConfig.verify(message.sagemaker_operator_config);
                    if (error)
                        return "sagemaker_operator_config." + error;
                }
            }
            if (message.aws_lambda_operator_config != null && message.hasOwnProperty("aws_lambda_operator_config")) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                {
                    let error = $root.bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify(message.aws_lambda_operator_config);
                    if (error)
                        return "aws_lambda_operator_config." + error;
                }
            }
            if (message.azure_functions_operator_config != null && message.hasOwnProperty("azure_functions_operator_config")) {
                if (properties.deployment_operator_config === 1)
                    return "deployment_operator_config: multiple values";
                properties.deployment_operator_config = 1;
                {
                    let error = $root.bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify(message.azure_functions_operator_config);
                    if (error)
                        return "azure_functions_operator_config." + error;
                }
            }
            return null;
        };

        /**
         * Creates a DeploymentSpec message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DeploymentSpec
         * @static
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
         * Creates a plain object from a DeploymentSpec message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DeploymentSpec
         * @static
         * @param {bentoml.DeploymentSpec} message DeploymentSpec
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DeploymentSpec to JSON.
         * @function toJSON
         * @memberof bentoml.DeploymentSpec
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        DeploymentSpec.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * DeploymentOperator enum.
         * @name bentoml.DeploymentSpec.DeploymentOperator
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
             * @memberof bentoml.DeploymentSpec
             * @interface ICustomOperatorConfig
             * @property {string|null} [name] CustomOperatorConfig name
             * @property {google.protobuf.IStruct|null} [config] CustomOperatorConfig config
             */

            /**
             * Constructs a new CustomOperatorConfig.
             * @memberof bentoml.DeploymentSpec
             * @classdesc Represents a CustomOperatorConfig.
             * @implements ICustomOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.ICustomOperatorConfig=} [properties] Properties to set
             */
            function CustomOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CustomOperatorConfig name.
             * @member {string} name
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @instance
             */
            CustomOperatorConfig.prototype.name = "";

            /**
             * CustomOperatorConfig config.
             * @member {google.protobuf.IStruct|null|undefined} config
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @instance
             */
            CustomOperatorConfig.prototype.config = null;

            /**
             * Creates a new CustomOperatorConfig instance using the specified properties.
             * @function create
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ICustomOperatorConfig=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig instance
             */
            CustomOperatorConfig.create = function create(properties) {
                return new CustomOperatorConfig(properties);
            };

            /**
             * Encodes the specified CustomOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @function encode
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ICustomOperatorConfig} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CustomOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.config != null && Object.hasOwnProperty.call(message, "config"))
                    $root.google.protobuf.Struct.encode(message.config, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified CustomOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.CustomOperatorConfig.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ICustomOperatorConfig} message CustomOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CustomOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a CustomOperatorConfig message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.CustomOperatorConfig} CustomOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CustomOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a CustomOperatorConfig message.
             * @function verify
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            CustomOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.config != null && message.hasOwnProperty("config")) {
                    let error = $root.google.protobuf.Struct.verify(message.config);
                    if (error)
                        return "config." + error;
                }
                return null;
            };

            /**
             * Creates a CustomOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
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
             * Creates a plain object from a CustomOperatorConfig message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.CustomOperatorConfig} message CustomOperatorConfig
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this CustomOperatorConfig to JSON.
             * @function toJSON
             * @memberof bentoml.DeploymentSpec.CustomOperatorConfig
             * @instance
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
             * @memberof bentoml.DeploymentSpec
             * @interface ISageMakerOperatorConfig
             * @property {string|null} [region] SageMakerOperatorConfig region
             * @property {string|null} [instance_type] SageMakerOperatorConfig instance_type
             * @property {number|null} [instance_count] SageMakerOperatorConfig instance_count
             * @property {string|null} [api_name] SageMakerOperatorConfig api_name
             * @property {number|null} [num_of_gunicorn_workers_per_instance] SageMakerOperatorConfig num_of_gunicorn_workers_per_instance
             * @property {number|null} [timeout] SageMakerOperatorConfig timeout
             */

            /**
             * Constructs a new SageMakerOperatorConfig.
             * @memberof bentoml.DeploymentSpec
             * @classdesc Represents a SageMakerOperatorConfig.
             * @implements ISageMakerOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.ISageMakerOperatorConfig=} [properties] Properties to set
             */
            function SageMakerOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SageMakerOperatorConfig region.
             * @member {string} region
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.region = "";

            /**
             * SageMakerOperatorConfig instance_type.
             * @member {string} instance_type
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.instance_type = "";

            /**
             * SageMakerOperatorConfig instance_count.
             * @member {number} instance_count
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.instance_count = 0;

            /**
             * SageMakerOperatorConfig api_name.
             * @member {string} api_name
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.api_name = "";

            /**
             * SageMakerOperatorConfig num_of_gunicorn_workers_per_instance.
             * @member {number} num_of_gunicorn_workers_per_instance
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.num_of_gunicorn_workers_per_instance = 0;

            /**
             * SageMakerOperatorConfig timeout.
             * @member {number} timeout
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
             */
            SageMakerOperatorConfig.prototype.timeout = 0;

            /**
             * Creates a new SageMakerOperatorConfig instance using the specified properties.
             * @function create
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ISageMakerOperatorConfig=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig instance
             */
            SageMakerOperatorConfig.create = function create(properties) {
                return new SageMakerOperatorConfig(properties);
            };

            /**
             * Encodes the specified SageMakerOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @function encode
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ISageMakerOperatorConfig} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SageMakerOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.region != null && Object.hasOwnProperty.call(message, "region"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.region);
                if (message.instance_type != null && Object.hasOwnProperty.call(message, "instance_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.instance_type);
                if (message.instance_count != null && Object.hasOwnProperty.call(message, "instance_count"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.instance_count);
                if (message.api_name != null && Object.hasOwnProperty.call(message, "api_name"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.api_name);
                if (message.num_of_gunicorn_workers_per_instance != null && Object.hasOwnProperty.call(message, "num_of_gunicorn_workers_per_instance"))
                    writer.uint32(/* id 5, wireType 0 =*/40).int32(message.num_of_gunicorn_workers_per_instance);
                if (message.timeout != null && Object.hasOwnProperty.call(message, "timeout"))
                    writer.uint32(/* id 6, wireType 0 =*/48).int32(message.timeout);
                return writer;
            };

            /**
             * Encodes the specified SageMakerOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.SageMakerOperatorConfig.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.ISageMakerOperatorConfig} message SageMakerOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SageMakerOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a SageMakerOperatorConfig message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.SageMakerOperatorConfig} SageMakerOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SageMakerOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a SageMakerOperatorConfig message.
             * @function verify
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            SageMakerOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.region != null && message.hasOwnProperty("region"))
                    if (!$util.isString(message.region))
                        return "region: string expected";
                if (message.instance_type != null && message.hasOwnProperty("instance_type"))
                    if (!$util.isString(message.instance_type))
                        return "instance_type: string expected";
                if (message.instance_count != null && message.hasOwnProperty("instance_count"))
                    if (!$util.isInteger(message.instance_count))
                        return "instance_count: integer expected";
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    if (!$util.isString(message.api_name))
                        return "api_name: string expected";
                if (message.num_of_gunicorn_workers_per_instance != null && message.hasOwnProperty("num_of_gunicorn_workers_per_instance"))
                    if (!$util.isInteger(message.num_of_gunicorn_workers_per_instance))
                        return "num_of_gunicorn_workers_per_instance: integer expected";
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    if (!$util.isInteger(message.timeout))
                        return "timeout: integer expected";
                return null;
            };

            /**
             * Creates a SageMakerOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
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
             * Creates a plain object from a SageMakerOperatorConfig message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.SageMakerOperatorConfig} message SageMakerOperatorConfig
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this SageMakerOperatorConfig to JSON.
             * @function toJSON
             * @memberof bentoml.DeploymentSpec.SageMakerOperatorConfig
             * @instance
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
             * @memberof bentoml.DeploymentSpec
             * @interface IAwsLambdaOperatorConfig
             * @property {string|null} [region] AwsLambdaOperatorConfig region
             * @property {string|null} [api_name] AwsLambdaOperatorConfig api_name
             * @property {number|null} [memory_size] AwsLambdaOperatorConfig memory_size
             * @property {number|null} [timeout] AwsLambdaOperatorConfig timeout
             */

            /**
             * Constructs a new AwsLambdaOperatorConfig.
             * @memberof bentoml.DeploymentSpec
             * @classdesc Represents an AwsLambdaOperatorConfig.
             * @implements IAwsLambdaOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig=} [properties] Properties to set
             */
            function AwsLambdaOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AwsLambdaOperatorConfig region.
             * @member {string} region
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @instance
             */
            AwsLambdaOperatorConfig.prototype.region = "";

            /**
             * AwsLambdaOperatorConfig api_name.
             * @member {string} api_name
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @instance
             */
            AwsLambdaOperatorConfig.prototype.api_name = "";

            /**
             * AwsLambdaOperatorConfig memory_size.
             * @member {number} memory_size
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @instance
             */
            AwsLambdaOperatorConfig.prototype.memory_size = 0;

            /**
             * AwsLambdaOperatorConfig timeout.
             * @member {number} timeout
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @instance
             */
            AwsLambdaOperatorConfig.prototype.timeout = 0;

            /**
             * Creates a new AwsLambdaOperatorConfig instance using the specified properties.
             * @function create
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig instance
             */
            AwsLambdaOperatorConfig.create = function create(properties) {
                return new AwsLambdaOperatorConfig(properties);
            };

            /**
             * Encodes the specified AwsLambdaOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @function encode
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AwsLambdaOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.region != null && Object.hasOwnProperty.call(message, "region"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.region);
                if (message.api_name != null && Object.hasOwnProperty.call(message, "api_name"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.api_name);
                if (message.memory_size != null && Object.hasOwnProperty.call(message, "memory_size"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.memory_size);
                if (message.timeout != null && Object.hasOwnProperty.call(message, "timeout"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.timeout);
                return writer;
            };

            /**
             * Encodes the specified AwsLambdaOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AwsLambdaOperatorConfig.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAwsLambdaOperatorConfig} message AwsLambdaOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AwsLambdaOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AwsLambdaOperatorConfig message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} AwsLambdaOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AwsLambdaOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AwsLambdaOperatorConfig message.
             * @function verify
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AwsLambdaOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.region != null && message.hasOwnProperty("region"))
                    if (!$util.isString(message.region))
                        return "region: string expected";
                if (message.api_name != null && message.hasOwnProperty("api_name"))
                    if (!$util.isString(message.api_name))
                        return "api_name: string expected";
                if (message.memory_size != null && message.hasOwnProperty("memory_size"))
                    if (!$util.isInteger(message.memory_size))
                        return "memory_size: integer expected";
                if (message.timeout != null && message.hasOwnProperty("timeout"))
                    if (!$util.isInteger(message.timeout))
                        return "timeout: integer expected";
                return null;
            };

            /**
             * Creates an AwsLambdaOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
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
             * Creates a plain object from an AwsLambdaOperatorConfig message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.AwsLambdaOperatorConfig} message AwsLambdaOperatorConfig
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this AwsLambdaOperatorConfig to JSON.
             * @function toJSON
             * @memberof bentoml.DeploymentSpec.AwsLambdaOperatorConfig
             * @instance
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
             * @memberof bentoml.DeploymentSpec
             * @interface IAzureFunctionsOperatorConfig
             * @property {string|null} [location] AzureFunctionsOperatorConfig location
             * @property {string|null} [premium_plan_sku] AzureFunctionsOperatorConfig premium_plan_sku
             * @property {number|null} [min_instances] AzureFunctionsOperatorConfig min_instances
             * @property {number|null} [max_burst] AzureFunctionsOperatorConfig max_burst
             * @property {string|null} [function_auth_level] AzureFunctionsOperatorConfig function_auth_level
             */

            /**
             * Constructs a new AzureFunctionsOperatorConfig.
             * @memberof bentoml.DeploymentSpec
             * @classdesc Represents an AzureFunctionsOperatorConfig.
             * @implements IAzureFunctionsOperatorConfig
             * @constructor
             * @param {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig=} [properties] Properties to set
             */
            function AzureFunctionsOperatorConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AzureFunctionsOperatorConfig location.
             * @member {string} location
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
             */
            AzureFunctionsOperatorConfig.prototype.location = "";

            /**
             * AzureFunctionsOperatorConfig premium_plan_sku.
             * @member {string} premium_plan_sku
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
             */
            AzureFunctionsOperatorConfig.prototype.premium_plan_sku = "";

            /**
             * AzureFunctionsOperatorConfig min_instances.
             * @member {number} min_instances
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
             */
            AzureFunctionsOperatorConfig.prototype.min_instances = 0;

            /**
             * AzureFunctionsOperatorConfig max_burst.
             * @member {number} max_burst
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
             */
            AzureFunctionsOperatorConfig.prototype.max_burst = 0;

            /**
             * AzureFunctionsOperatorConfig function_auth_level.
             * @member {string} function_auth_level
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
             */
            AzureFunctionsOperatorConfig.prototype.function_auth_level = "";

            /**
             * Creates a new AzureFunctionsOperatorConfig instance using the specified properties.
             * @function create
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig=} [properties] Properties to set
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig instance
             */
            AzureFunctionsOperatorConfig.create = function create(properties) {
                return new AzureFunctionsOperatorConfig(properties);
            };

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @function encode
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AzureFunctionsOperatorConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.location != null && Object.hasOwnProperty.call(message, "location"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.location);
                if (message.premium_plan_sku != null && Object.hasOwnProperty.call(message, "premium_plan_sku"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.premium_plan_sku);
                if (message.min_instances != null && Object.hasOwnProperty.call(message, "min_instances"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.min_instances);
                if (message.max_burst != null && Object.hasOwnProperty.call(message, "max_burst"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.max_burst);
                if (message.function_auth_level != null && Object.hasOwnProperty.call(message, "function_auth_level"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.function_auth_level);
                return writer;
            };

            /**
             * Encodes the specified AzureFunctionsOperatorConfig message, length delimited. Does not implicitly {@link bentoml.DeploymentSpec.AzureFunctionsOperatorConfig.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.IAzureFunctionsOperatorConfig} message AzureFunctionsOperatorConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AzureFunctionsOperatorConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AzureFunctionsOperatorConfig message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} AzureFunctionsOperatorConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AzureFunctionsOperatorConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AzureFunctionsOperatorConfig message.
             * @function verify
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AzureFunctionsOperatorConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.location != null && message.hasOwnProperty("location"))
                    if (!$util.isString(message.location))
                        return "location: string expected";
                if (message.premium_plan_sku != null && message.hasOwnProperty("premium_plan_sku"))
                    if (!$util.isString(message.premium_plan_sku))
                        return "premium_plan_sku: string expected";
                if (message.min_instances != null && message.hasOwnProperty("min_instances"))
                    if (!$util.isInteger(message.min_instances))
                        return "min_instances: integer expected";
                if (message.max_burst != null && message.hasOwnProperty("max_burst"))
                    if (!$util.isInteger(message.max_burst))
                        return "max_burst: integer expected";
                if (message.function_auth_level != null && message.hasOwnProperty("function_auth_level"))
                    if (!$util.isString(message.function_auth_level))
                        return "function_auth_level: string expected";
                return null;
            };

            /**
             * Creates an AzureFunctionsOperatorConfig message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
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
             * Creates a plain object from an AzureFunctionsOperatorConfig message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @static
             * @param {bentoml.DeploymentSpec.AzureFunctionsOperatorConfig} message AzureFunctionsOperatorConfig
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this AzureFunctionsOperatorConfig to JSON.
             * @function toJSON
             * @memberof bentoml.DeploymentSpec.AzureFunctionsOperatorConfig
             * @instance
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
         * @memberof bentoml
         * @interface IDeploymentState
         * @property {bentoml.DeploymentState.State|null} [state] DeploymentState state
         * @property {string|null} [error_message] DeploymentState error_message
         * @property {string|null} [info_json] DeploymentState info_json
         * @property {google.protobuf.ITimestamp|null} [timestamp] DeploymentState timestamp
         */

        /**
         * Constructs a new DeploymentState.
         * @memberof bentoml
         * @classdesc Represents a DeploymentState.
         * @implements IDeploymentState
         * @constructor
         * @param {bentoml.IDeploymentState=} [properties] Properties to set
         */
        function DeploymentState(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentState state.
         * @member {bentoml.DeploymentState.State} state
         * @memberof bentoml.DeploymentState
         * @instance
         */
        DeploymentState.prototype.state = 0;

        /**
         * DeploymentState error_message.
         * @member {string} error_message
         * @memberof bentoml.DeploymentState
         * @instance
         */
        DeploymentState.prototype.error_message = "";

        /**
         * DeploymentState info_json.
         * @member {string} info_json
         * @memberof bentoml.DeploymentState
         * @instance
         */
        DeploymentState.prototype.info_json = "";

        /**
         * DeploymentState timestamp.
         * @member {google.protobuf.ITimestamp|null|undefined} timestamp
         * @memberof bentoml.DeploymentState
         * @instance
         */
        DeploymentState.prototype.timestamp = null;

        /**
         * Creates a new DeploymentState instance using the specified properties.
         * @function create
         * @memberof bentoml.DeploymentState
         * @static
         * @param {bentoml.IDeploymentState=} [properties] Properties to set
         * @returns {bentoml.DeploymentState} DeploymentState instance
         */
        DeploymentState.create = function create(properties) {
            return new DeploymentState(properties);
        };

        /**
         * Encodes the specified DeploymentState message. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DeploymentState
         * @static
         * @param {bentoml.IDeploymentState} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentState.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.state != null && Object.hasOwnProperty.call(message, "state"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.state);
            if (message.error_message != null && Object.hasOwnProperty.call(message, "error_message"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.error_message);
            if (message.info_json != null && Object.hasOwnProperty.call(message, "info_json"))
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.info_json);
            if (message.timestamp != null && Object.hasOwnProperty.call(message, "timestamp"))
                $root.google.protobuf.Timestamp.encode(message.timestamp, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentState message, length delimited. Does not implicitly {@link bentoml.DeploymentState.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DeploymentState
         * @static
         * @param {bentoml.IDeploymentState} message DeploymentState message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentState.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentState message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DeploymentState
         * @static
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
                    message.state = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.DeploymentState
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentState} DeploymentState
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentState.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentState message.
         * @function verify
         * @memberof bentoml.DeploymentState
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DeploymentState.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.state != null && message.hasOwnProperty("state"))
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
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            if (message.info_json != null && message.hasOwnProperty("info_json"))
                if (!$util.isString(message.info_json))
                    return "info_json: string expected";
            if (message.timestamp != null && message.hasOwnProperty("timestamp")) {
                let error = $root.google.protobuf.Timestamp.verify(message.timestamp);
                if (error)
                    return "timestamp." + error;
            }
            return null;
        };

        /**
         * Creates a DeploymentState message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DeploymentState
         * @static
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
         * Creates a plain object from a DeploymentState message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DeploymentState
         * @static
         * @param {bentoml.DeploymentState} message DeploymentState
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DeploymentState to JSON.
         * @function toJSON
         * @memberof bentoml.DeploymentState
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        DeploymentState.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * State enum.
         * @name bentoml.DeploymentState.State
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
         * @memberof bentoml
         * @interface IDeployment
         * @property {string|null} [namespace] Deployment namespace
         * @property {string|null} [name] Deployment name
         * @property {bentoml.IDeploymentSpec|null} [spec] Deployment spec
         * @property {bentoml.IDeploymentState|null} [state] Deployment state
         * @property {Object.<string,string>|null} [annotations] Deployment annotations
         * @property {Object.<string,string>|null} [labels] Deployment labels
         * @property {google.protobuf.ITimestamp|null} [created_at] Deployment created_at
         * @property {google.protobuf.ITimestamp|null} [last_updated_at] Deployment last_updated_at
         */

        /**
         * Constructs a new Deployment.
         * @memberof bentoml
         * @classdesc Represents a Deployment.
         * @implements IDeployment
         * @constructor
         * @param {bentoml.IDeployment=} [properties] Properties to set
         */
        function Deployment(properties) {
            this.annotations = {};
            this.labels = {};
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Deployment namespace.
         * @member {string} namespace
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.namespace = "";

        /**
         * Deployment name.
         * @member {string} name
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.name = "";

        /**
         * Deployment spec.
         * @member {bentoml.IDeploymentSpec|null|undefined} spec
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.spec = null;

        /**
         * Deployment state.
         * @member {bentoml.IDeploymentState|null|undefined} state
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.state = null;

        /**
         * Deployment annotations.
         * @member {Object.<string,string>} annotations
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.annotations = $util.emptyObject;

        /**
         * Deployment labels.
         * @member {Object.<string,string>} labels
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.labels = $util.emptyObject;

        /**
         * Deployment created_at.
         * @member {google.protobuf.ITimestamp|null|undefined} created_at
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.created_at = null;

        /**
         * Deployment last_updated_at.
         * @member {google.protobuf.ITimestamp|null|undefined} last_updated_at
         * @memberof bentoml.Deployment
         * @instance
         */
        Deployment.prototype.last_updated_at = null;

        /**
         * Creates a new Deployment instance using the specified properties.
         * @function create
         * @memberof bentoml.Deployment
         * @static
         * @param {bentoml.IDeployment=} [properties] Properties to set
         * @returns {bentoml.Deployment} Deployment instance
         */
        Deployment.create = function create(properties) {
            return new Deployment(properties);
        };

        /**
         * Encodes the specified Deployment message. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @function encode
         * @memberof bentoml.Deployment
         * @static
         * @param {bentoml.IDeployment} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Deployment.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.namespace != null && Object.hasOwnProperty.call(message, "namespace"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.namespace);
            if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.name);
            if (message.spec != null && Object.hasOwnProperty.call(message, "spec"))
                $root.bentoml.DeploymentSpec.encode(message.spec, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.state != null && Object.hasOwnProperty.call(message, "state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.annotations != null && Object.hasOwnProperty.call(message, "annotations"))
                for (let keys = Object.keys(message.annotations), i = 0; i < keys.length; ++i)
                    writer.uint32(/* id 5, wireType 2 =*/42).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.annotations[keys[i]]).ldelim();
            if (message.labels != null && Object.hasOwnProperty.call(message, "labels"))
                for (let keys = Object.keys(message.labels), i = 0; i < keys.length; ++i)
                    writer.uint32(/* id 6, wireType 2 =*/50).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.labels[keys[i]]).ldelim();
            if (message.created_at != null && Object.hasOwnProperty.call(message, "created_at"))
                $root.google.protobuf.Timestamp.encode(message.created_at, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
            if (message.last_updated_at != null && Object.hasOwnProperty.call(message, "last_updated_at"))
                $root.google.protobuf.Timestamp.encode(message.last_updated_at, writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified Deployment message, length delimited. Does not implicitly {@link bentoml.Deployment.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.Deployment
         * @static
         * @param {bentoml.IDeployment} message Deployment message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Deployment.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Deployment message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.Deployment
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.Deployment
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Deployment} Deployment
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Deployment.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Deployment message.
         * @function verify
         * @memberof bentoml.Deployment
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Deployment.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.name != null && message.hasOwnProperty("name"))
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.spec != null && message.hasOwnProperty("spec")) {
                let error = $root.bentoml.DeploymentSpec.verify(message.spec);
                if (error)
                    return "spec." + error;
            }
            if (message.state != null && message.hasOwnProperty("state")) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            if (message.annotations != null && message.hasOwnProperty("annotations")) {
                if (!$util.isObject(message.annotations))
                    return "annotations: object expected";
                let key = Object.keys(message.annotations);
                for (let i = 0; i < key.length; ++i)
                    if (!$util.isString(message.annotations[key[i]]))
                        return "annotations: string{k:string} expected";
            }
            if (message.labels != null && message.hasOwnProperty("labels")) {
                if (!$util.isObject(message.labels))
                    return "labels: object expected";
                let key = Object.keys(message.labels);
                for (let i = 0; i < key.length; ++i)
                    if (!$util.isString(message.labels[key[i]]))
                        return "labels: string{k:string} expected";
            }
            if (message.created_at != null && message.hasOwnProperty("created_at")) {
                let error = $root.google.protobuf.Timestamp.verify(message.created_at);
                if (error)
                    return "created_at." + error;
            }
            if (message.last_updated_at != null && message.hasOwnProperty("last_updated_at")) {
                let error = $root.google.protobuf.Timestamp.verify(message.last_updated_at);
                if (error)
                    return "last_updated_at." + error;
            }
            return null;
        };

        /**
         * Creates a Deployment message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.Deployment
         * @static
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
         * Creates a plain object from a Deployment message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.Deployment
         * @static
         * @param {bentoml.Deployment} message Deployment
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this Deployment to JSON.
         * @function toJSON
         * @memberof bentoml.Deployment
         * @instance
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
         * @memberof bentoml
         * @interface IDeploymentStatus
         * @property {bentoml.IDeploymentState|null} [state] DeploymentStatus state
         */

        /**
         * Constructs a new DeploymentStatus.
         * @memberof bentoml
         * @classdesc Represents a DeploymentStatus.
         * @implements IDeploymentStatus
         * @constructor
         * @param {bentoml.IDeploymentStatus=} [properties] Properties to set
         */
        function DeploymentStatus(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeploymentStatus state.
         * @member {bentoml.IDeploymentState|null|undefined} state
         * @memberof bentoml.DeploymentStatus
         * @instance
         */
        DeploymentStatus.prototype.state = null;

        /**
         * Creates a new DeploymentStatus instance using the specified properties.
         * @function create
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {bentoml.IDeploymentStatus=} [properties] Properties to set
         * @returns {bentoml.DeploymentStatus} DeploymentStatus instance
         */
        DeploymentStatus.create = function create(properties) {
            return new DeploymentStatus(properties);
        };

        /**
         * Encodes the specified DeploymentStatus message. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {bentoml.IDeploymentStatus} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentStatus.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.state != null && Object.hasOwnProperty.call(message, "state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeploymentStatus message, length delimited. Does not implicitly {@link bentoml.DeploymentStatus.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {bentoml.IDeploymentStatus} message DeploymentStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeploymentStatus.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeploymentStatus message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DeploymentStatus
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeploymentStatus} DeploymentStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeploymentStatus.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeploymentStatus message.
         * @function verify
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DeploymentStatus.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.state != null && message.hasOwnProperty("state")) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            return null;
        };

        /**
         * Creates a DeploymentStatus message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DeploymentStatus
         * @static
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
         * Creates a plain object from a DeploymentStatus message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DeploymentStatus
         * @static
         * @param {bentoml.DeploymentStatus} message DeploymentStatus
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DeploymentStatus to JSON.
         * @function toJSON
         * @memberof bentoml.DeploymentStatus
         * @instance
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
         * @memberof bentoml
         * @interface IApplyDeploymentRequest
         * @property {bentoml.IDeployment|null} [deployment] ApplyDeploymentRequest deployment
         */

        /**
         * Constructs a new ApplyDeploymentRequest.
         * @memberof bentoml
         * @classdesc Represents an ApplyDeploymentRequest.
         * @implements IApplyDeploymentRequest
         * @constructor
         * @param {bentoml.IApplyDeploymentRequest=} [properties] Properties to set
         */
        function ApplyDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ApplyDeploymentRequest deployment.
         * @member {bentoml.IDeployment|null|undefined} deployment
         * @memberof bentoml.ApplyDeploymentRequest
         * @instance
         */
        ApplyDeploymentRequest.prototype.deployment = null;

        /**
         * Creates a new ApplyDeploymentRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {bentoml.IApplyDeploymentRequest=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest instance
         */
        ApplyDeploymentRequest.create = function create(properties) {
            return new ApplyDeploymentRequest(properties);
        };

        /**
         * Encodes the specified ApplyDeploymentRequest message. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {bentoml.IApplyDeploymentRequest} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment != null && Object.hasOwnProperty.call(message, "deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ApplyDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {bentoml.IApplyDeploymentRequest} message ApplyDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an ApplyDeploymentRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentRequest} ApplyDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an ApplyDeploymentRequest message.
         * @function verify
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ApplyDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment != null && message.hasOwnProperty("deployment")) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates an ApplyDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
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
         * Creates a plain object from an ApplyDeploymentRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ApplyDeploymentRequest
         * @static
         * @param {bentoml.ApplyDeploymentRequest} message ApplyDeploymentRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ApplyDeploymentRequest to JSON.
         * @function toJSON
         * @memberof bentoml.ApplyDeploymentRequest
         * @instance
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
         * @memberof bentoml
         * @interface IApplyDeploymentResponse
         * @property {bentoml.IStatus|null} [status] ApplyDeploymentResponse status
         * @property {bentoml.IDeployment|null} [deployment] ApplyDeploymentResponse deployment
         */

        /**
         * Constructs a new ApplyDeploymentResponse.
         * @memberof bentoml
         * @classdesc Represents an ApplyDeploymentResponse.
         * @implements IApplyDeploymentResponse
         * @constructor
         * @param {bentoml.IApplyDeploymentResponse=} [properties] Properties to set
         */
        function ApplyDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ApplyDeploymentResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.ApplyDeploymentResponse
         * @instance
         */
        ApplyDeploymentResponse.prototype.status = null;

        /**
         * ApplyDeploymentResponse deployment.
         * @member {bentoml.IDeployment|null|undefined} deployment
         * @memberof bentoml.ApplyDeploymentResponse
         * @instance
         */
        ApplyDeploymentResponse.prototype.deployment = null;

        /**
         * Creates a new ApplyDeploymentResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {bentoml.IApplyDeploymentResponse=} [properties] Properties to set
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse instance
         */
        ApplyDeploymentResponse.create = function create(properties) {
            return new ApplyDeploymentResponse(properties);
        };

        /**
         * Encodes the specified ApplyDeploymentResponse message. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {bentoml.IApplyDeploymentResponse} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployment != null && Object.hasOwnProperty.call(message, "deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ApplyDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.ApplyDeploymentResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {bentoml.IApplyDeploymentResponse} message ApplyDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ApplyDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an ApplyDeploymentResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ApplyDeploymentResponse} ApplyDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ApplyDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an ApplyDeploymentResponse message.
         * @function verify
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ApplyDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployment != null && message.hasOwnProperty("deployment")) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates an ApplyDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
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
         * Creates a plain object from an ApplyDeploymentResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ApplyDeploymentResponse
         * @static
         * @param {bentoml.ApplyDeploymentResponse} message ApplyDeploymentResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ApplyDeploymentResponse to JSON.
         * @function toJSON
         * @memberof bentoml.ApplyDeploymentResponse
         * @instance
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
         * @memberof bentoml
         * @interface IDeleteDeploymentRequest
         * @property {string|null} [deployment_name] DeleteDeploymentRequest deployment_name
         * @property {string|null} [namespace] DeleteDeploymentRequest namespace
         * @property {boolean|null} [force_delete] DeleteDeploymentRequest force_delete
         */

        /**
         * Constructs a new DeleteDeploymentRequest.
         * @memberof bentoml
         * @classdesc Represents a DeleteDeploymentRequest.
         * @implements IDeleteDeploymentRequest
         * @constructor
         * @param {bentoml.IDeleteDeploymentRequest=} [properties] Properties to set
         */
        function DeleteDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeleteDeploymentRequest deployment_name.
         * @member {string} deployment_name
         * @memberof bentoml.DeleteDeploymentRequest
         * @instance
         */
        DeleteDeploymentRequest.prototype.deployment_name = "";

        /**
         * DeleteDeploymentRequest namespace.
         * @member {string} namespace
         * @memberof bentoml.DeleteDeploymentRequest
         * @instance
         */
        DeleteDeploymentRequest.prototype.namespace = "";

        /**
         * DeleteDeploymentRequest force_delete.
         * @member {boolean} force_delete
         * @memberof bentoml.DeleteDeploymentRequest
         * @instance
         */
        DeleteDeploymentRequest.prototype.force_delete = false;

        /**
         * Creates a new DeleteDeploymentRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {bentoml.IDeleteDeploymentRequest=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest instance
         */
        DeleteDeploymentRequest.create = function create(properties) {
            return new DeleteDeploymentRequest(properties);
        };

        /**
         * Encodes the specified DeleteDeploymentRequest message. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {bentoml.IDeleteDeploymentRequest} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && Object.hasOwnProperty.call(message, "deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && Object.hasOwnProperty.call(message, "namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            if (message.force_delete != null && Object.hasOwnProperty.call(message, "force_delete"))
                writer.uint32(/* id 3, wireType 0 =*/24).bool(message.force_delete);
            return writer;
        };

        /**
         * Encodes the specified DeleteDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {bentoml.IDeleteDeploymentRequest} message DeleteDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeleteDeploymentRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentRequest} DeleteDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeleteDeploymentRequest message.
         * @function verify
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DeleteDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.force_delete != null && message.hasOwnProperty("force_delete"))
                if (typeof message.force_delete !== "boolean")
                    return "force_delete: boolean expected";
            return null;
        };

        /**
         * Creates a DeleteDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
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
         * Creates a plain object from a DeleteDeploymentRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DeleteDeploymentRequest
         * @static
         * @param {bentoml.DeleteDeploymentRequest} message DeleteDeploymentRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DeleteDeploymentRequest to JSON.
         * @function toJSON
         * @memberof bentoml.DeleteDeploymentRequest
         * @instance
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
         * @memberof bentoml
         * @interface IDeleteDeploymentResponse
         * @property {bentoml.IStatus|null} [status] DeleteDeploymentResponse status
         */

        /**
         * Constructs a new DeleteDeploymentResponse.
         * @memberof bentoml
         * @classdesc Represents a DeleteDeploymentResponse.
         * @implements IDeleteDeploymentResponse
         * @constructor
         * @param {bentoml.IDeleteDeploymentResponse=} [properties] Properties to set
         */
        function DeleteDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DeleteDeploymentResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.DeleteDeploymentResponse
         * @instance
         */
        DeleteDeploymentResponse.prototype.status = null;

        /**
         * Creates a new DeleteDeploymentResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {bentoml.IDeleteDeploymentResponse=} [properties] Properties to set
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse instance
         */
        DeleteDeploymentResponse.create = function create(properties) {
            return new DeleteDeploymentResponse(properties);
        };

        /**
         * Encodes the specified DeleteDeploymentResponse message. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {bentoml.IDeleteDeploymentResponse} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DeleteDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DeleteDeploymentResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {bentoml.IDeleteDeploymentResponse} message DeleteDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DeleteDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DeleteDeploymentResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DeleteDeploymentResponse} DeleteDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DeleteDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DeleteDeploymentResponse message.
         * @function verify
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DeleteDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a DeleteDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
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
         * Creates a plain object from a DeleteDeploymentResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DeleteDeploymentResponse
         * @static
         * @param {bentoml.DeleteDeploymentResponse} message DeleteDeploymentResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DeleteDeploymentResponse to JSON.
         * @function toJSON
         * @memberof bentoml.DeleteDeploymentResponse
         * @instance
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
         * @memberof bentoml
         * @interface IGetDeploymentRequest
         * @property {string|null} [deployment_name] GetDeploymentRequest deployment_name
         * @property {string|null} [namespace] GetDeploymentRequest namespace
         */

        /**
         * Constructs a new GetDeploymentRequest.
         * @memberof bentoml
         * @classdesc Represents a GetDeploymentRequest.
         * @implements IGetDeploymentRequest
         * @constructor
         * @param {bentoml.IGetDeploymentRequest=} [properties] Properties to set
         */
        function GetDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetDeploymentRequest deployment_name.
         * @member {string} deployment_name
         * @memberof bentoml.GetDeploymentRequest
         * @instance
         */
        GetDeploymentRequest.prototype.deployment_name = "";

        /**
         * GetDeploymentRequest namespace.
         * @member {string} namespace
         * @memberof bentoml.GetDeploymentRequest
         * @instance
         */
        GetDeploymentRequest.prototype.namespace = "";

        /**
         * Creates a new GetDeploymentRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {bentoml.IGetDeploymentRequest=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest instance
         */
        GetDeploymentRequest.create = function create(properties) {
            return new GetDeploymentRequest(properties);
        };

        /**
         * Encodes the specified GetDeploymentRequest message. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {bentoml.IGetDeploymentRequest} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && Object.hasOwnProperty.call(message, "deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && Object.hasOwnProperty.call(message, "namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            return writer;
        };

        /**
         * Encodes the specified GetDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.GetDeploymentRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {bentoml.IGetDeploymentRequest} message GetDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetDeploymentRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.GetDeploymentRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentRequest} GetDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetDeploymentRequest message.
         * @function verify
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GetDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            return null;
        };

        /**
         * Creates a GetDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.GetDeploymentRequest
         * @static
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
         * Creates a plain object from a GetDeploymentRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.GetDeploymentRequest
         * @static
         * @param {bentoml.GetDeploymentRequest} message GetDeploymentRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this GetDeploymentRequest to JSON.
         * @function toJSON
         * @memberof bentoml.GetDeploymentRequest
         * @instance
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
         * @memberof bentoml
         * @interface IGetDeploymentResponse
         * @property {bentoml.IStatus|null} [status] GetDeploymentResponse status
         * @property {bentoml.IDeployment|null} [deployment] GetDeploymentResponse deployment
         */

        /**
         * Constructs a new GetDeploymentResponse.
         * @memberof bentoml
         * @classdesc Represents a GetDeploymentResponse.
         * @implements IGetDeploymentResponse
         * @constructor
         * @param {bentoml.IGetDeploymentResponse=} [properties] Properties to set
         */
        function GetDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetDeploymentResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.GetDeploymentResponse
         * @instance
         */
        GetDeploymentResponse.prototype.status = null;

        /**
         * GetDeploymentResponse deployment.
         * @member {bentoml.IDeployment|null|undefined} deployment
         * @memberof bentoml.GetDeploymentResponse
         * @instance
         */
        GetDeploymentResponse.prototype.deployment = null;

        /**
         * Creates a new GetDeploymentResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {bentoml.IGetDeploymentResponse=} [properties] Properties to set
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse instance
         */
        GetDeploymentResponse.create = function create(properties) {
            return new GetDeploymentResponse(properties);
        };

        /**
         * Encodes the specified GetDeploymentResponse message. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {bentoml.IGetDeploymentResponse} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployment != null && Object.hasOwnProperty.call(message, "deployment"))
                $root.bentoml.Deployment.encode(message.deployment, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified GetDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.GetDeploymentResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {bentoml.IGetDeploymentResponse} message GetDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetDeploymentResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.GetDeploymentResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetDeploymentResponse} GetDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetDeploymentResponse message.
         * @function verify
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GetDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployment != null && message.hasOwnProperty("deployment")) {
                let error = $root.bentoml.Deployment.verify(message.deployment);
                if (error)
                    return "deployment." + error;
            }
            return null;
        };

        /**
         * Creates a GetDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.GetDeploymentResponse
         * @static
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
         * Creates a plain object from a GetDeploymentResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.GetDeploymentResponse
         * @static
         * @param {bentoml.GetDeploymentResponse} message GetDeploymentResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this GetDeploymentResponse to JSON.
         * @function toJSON
         * @memberof bentoml.GetDeploymentResponse
         * @instance
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
         * @memberof bentoml
         * @interface IDescribeDeploymentRequest
         * @property {string|null} [deployment_name] DescribeDeploymentRequest deployment_name
         * @property {string|null} [namespace] DescribeDeploymentRequest namespace
         */

        /**
         * Constructs a new DescribeDeploymentRequest.
         * @memberof bentoml
         * @classdesc Represents a DescribeDeploymentRequest.
         * @implements IDescribeDeploymentRequest
         * @constructor
         * @param {bentoml.IDescribeDeploymentRequest=} [properties] Properties to set
         */
        function DescribeDeploymentRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DescribeDeploymentRequest deployment_name.
         * @member {string} deployment_name
         * @memberof bentoml.DescribeDeploymentRequest
         * @instance
         */
        DescribeDeploymentRequest.prototype.deployment_name = "";

        /**
         * DescribeDeploymentRequest namespace.
         * @member {string} namespace
         * @memberof bentoml.DescribeDeploymentRequest
         * @instance
         */
        DescribeDeploymentRequest.prototype.namespace = "";

        /**
         * Creates a new DescribeDeploymentRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {bentoml.IDescribeDeploymentRequest=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest instance
         */
        DescribeDeploymentRequest.create = function create(properties) {
            return new DescribeDeploymentRequest(properties);
        };

        /**
         * Encodes the specified DescribeDeploymentRequest message. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {bentoml.IDescribeDeploymentRequest} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.deployment_name != null && Object.hasOwnProperty.call(message, "deployment_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.deployment_name);
            if (message.namespace != null && Object.hasOwnProperty.call(message, "namespace"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.namespace);
            return writer;
        };

        /**
         * Encodes the specified DescribeDeploymentRequest message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {bentoml.IDescribeDeploymentRequest} message DescribeDeploymentRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DescribeDeploymentRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentRequest} DescribeDeploymentRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DescribeDeploymentRequest message.
         * @function verify
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DescribeDeploymentRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.deployment_name != null && message.hasOwnProperty("deployment_name"))
                if (!$util.isString(message.deployment_name))
                    return "deployment_name: string expected";
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            return null;
        };

        /**
         * Creates a DescribeDeploymentRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
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
         * Creates a plain object from a DescribeDeploymentRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DescribeDeploymentRequest
         * @static
         * @param {bentoml.DescribeDeploymentRequest} message DescribeDeploymentRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DescribeDeploymentRequest to JSON.
         * @function toJSON
         * @memberof bentoml.DescribeDeploymentRequest
         * @instance
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
         * @memberof bentoml
         * @interface IDescribeDeploymentResponse
         * @property {bentoml.IStatus|null} [status] DescribeDeploymentResponse status
         * @property {bentoml.IDeploymentState|null} [state] DescribeDeploymentResponse state
         */

        /**
         * Constructs a new DescribeDeploymentResponse.
         * @memberof bentoml
         * @classdesc Represents a DescribeDeploymentResponse.
         * @implements IDescribeDeploymentResponse
         * @constructor
         * @param {bentoml.IDescribeDeploymentResponse=} [properties] Properties to set
         */
        function DescribeDeploymentResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DescribeDeploymentResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.DescribeDeploymentResponse
         * @instance
         */
        DescribeDeploymentResponse.prototype.status = null;

        /**
         * DescribeDeploymentResponse state.
         * @member {bentoml.IDeploymentState|null|undefined} state
         * @memberof bentoml.DescribeDeploymentResponse
         * @instance
         */
        DescribeDeploymentResponse.prototype.state = null;

        /**
         * Creates a new DescribeDeploymentResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {bentoml.IDescribeDeploymentResponse=} [properties] Properties to set
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse instance
         */
        DescribeDeploymentResponse.create = function create(properties) {
            return new DescribeDeploymentResponse(properties);
        };

        /**
         * Encodes the specified DescribeDeploymentResponse message. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {bentoml.IDescribeDeploymentResponse} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.state != null && Object.hasOwnProperty.call(message, "state"))
                $root.bentoml.DeploymentState.encode(message.state, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DescribeDeploymentResponse message, length delimited. Does not implicitly {@link bentoml.DescribeDeploymentResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {bentoml.IDescribeDeploymentResponse} message DescribeDeploymentResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DescribeDeploymentResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DescribeDeploymentResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DescribeDeploymentResponse} DescribeDeploymentResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DescribeDeploymentResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DescribeDeploymentResponse message.
         * @function verify
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DescribeDeploymentResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.state != null && message.hasOwnProperty("state")) {
                let error = $root.bentoml.DeploymentState.verify(message.state);
                if (error)
                    return "state." + error;
            }
            return null;
        };

        /**
         * Creates a DescribeDeploymentResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
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
         * Creates a plain object from a DescribeDeploymentResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DescribeDeploymentResponse
         * @static
         * @param {bentoml.DescribeDeploymentResponse} message DescribeDeploymentResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DescribeDeploymentResponse to JSON.
         * @function toJSON
         * @memberof bentoml.DescribeDeploymentResponse
         * @instance
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
         * @memberof bentoml
         * @interface IListDeploymentsRequest
         * @property {string|null} [namespace] ListDeploymentsRequest namespace
         * @property {number|null} [offset] ListDeploymentsRequest offset
         * @property {number|null} [limit] ListDeploymentsRequest limit
         * @property {bentoml.DeploymentSpec.DeploymentOperator|null} [operator] ListDeploymentsRequest operator
         * @property {bentoml.ListDeploymentsRequest.SORTABLE_COLUMN|null} [order_by] ListDeploymentsRequest order_by
         * @property {boolean|null} [ascending_order] ListDeploymentsRequest ascending_order
         * @property {string|null} [labels_query] ListDeploymentsRequest labels_query
         */

        /**
         * Constructs a new ListDeploymentsRequest.
         * @memberof bentoml
         * @classdesc Represents a ListDeploymentsRequest.
         * @implements IListDeploymentsRequest
         * @constructor
         * @param {bentoml.IListDeploymentsRequest=} [properties] Properties to set
         */
        function ListDeploymentsRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListDeploymentsRequest namespace.
         * @member {string} namespace
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.namespace = "";

        /**
         * ListDeploymentsRequest offset.
         * @member {number} offset
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.offset = 0;

        /**
         * ListDeploymentsRequest limit.
         * @member {number} limit
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.limit = 0;

        /**
         * ListDeploymentsRequest operator.
         * @member {bentoml.DeploymentSpec.DeploymentOperator} operator
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.operator = 0;

        /**
         * ListDeploymentsRequest order_by.
         * @member {bentoml.ListDeploymentsRequest.SORTABLE_COLUMN} order_by
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.order_by = 0;

        /**
         * ListDeploymentsRequest ascending_order.
         * @member {boolean} ascending_order
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.ascending_order = false;

        /**
         * ListDeploymentsRequest labels_query.
         * @member {string} labels_query
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         */
        ListDeploymentsRequest.prototype.labels_query = "";

        /**
         * Creates a new ListDeploymentsRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {bentoml.IListDeploymentsRequest=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest instance
         */
        ListDeploymentsRequest.create = function create(properties) {
            return new ListDeploymentsRequest(properties);
        };

        /**
         * Encodes the specified ListDeploymentsRequest message. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {bentoml.IListDeploymentsRequest} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.namespace != null && Object.hasOwnProperty.call(message, "namespace"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.namespace);
            if (message.offset != null && Object.hasOwnProperty.call(message, "offset"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.offset);
            if (message.limit != null && Object.hasOwnProperty.call(message, "limit"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.limit);
            if (message.operator != null && Object.hasOwnProperty.call(message, "operator"))
                writer.uint32(/* id 4, wireType 0 =*/32).int32(message.operator);
            if (message.order_by != null && Object.hasOwnProperty.call(message, "order_by"))
                writer.uint32(/* id 5, wireType 0 =*/40).int32(message.order_by);
            if (message.ascending_order != null && Object.hasOwnProperty.call(message, "ascending_order"))
                writer.uint32(/* id 6, wireType 0 =*/48).bool(message.ascending_order);
            if (message.labels_query != null && Object.hasOwnProperty.call(message, "labels_query"))
                writer.uint32(/* id 7, wireType 2 =*/58).string(message.labels_query);
            return writer;
        };

        /**
         * Encodes the specified ListDeploymentsRequest message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {bentoml.IListDeploymentsRequest} message ListDeploymentsRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListDeploymentsRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ListDeploymentsRequest
         * @static
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
                    message.operator = reader.int32();
                    break;
                case 5:
                    message.order_by = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsRequest} ListDeploymentsRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListDeploymentsRequest message.
         * @function verify
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ListDeploymentsRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.namespace != null && message.hasOwnProperty("namespace"))
                if (!$util.isString(message.namespace))
                    return "namespace: string expected";
            if (message.offset != null && message.hasOwnProperty("offset"))
                if (!$util.isInteger(message.offset))
                    return "offset: integer expected";
            if (message.limit != null && message.hasOwnProperty("limit"))
                if (!$util.isInteger(message.limit))
                    return "limit: integer expected";
            if (message.operator != null && message.hasOwnProperty("operator"))
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
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                switch (message.order_by) {
                default:
                    return "order_by: enum value expected";
                case 0:
                case 1:
                    break;
                }
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                if (typeof message.ascending_order !== "boolean")
                    return "ascending_order: boolean expected";
            if (message.labels_query != null && message.hasOwnProperty("labels_query"))
                if (!$util.isString(message.labels_query))
                    return "labels_query: string expected";
            return null;
        };

        /**
         * Creates a ListDeploymentsRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.ListDeploymentsRequest
         * @static
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
         * Creates a plain object from a ListDeploymentsRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ListDeploymentsRequest
         * @static
         * @param {bentoml.ListDeploymentsRequest} message ListDeploymentsRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ListDeploymentsRequest to JSON.
         * @function toJSON
         * @memberof bentoml.ListDeploymentsRequest
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        ListDeploymentsRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * SORTABLE_COLUMN enum.
         * @name bentoml.ListDeploymentsRequest.SORTABLE_COLUMN
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
         * @memberof bentoml
         * @interface IListDeploymentsResponse
         * @property {bentoml.IStatus|null} [status] ListDeploymentsResponse status
         * @property {Array.<bentoml.IDeployment>|null} [deployments] ListDeploymentsResponse deployments
         */

        /**
         * Constructs a new ListDeploymentsResponse.
         * @memberof bentoml
         * @classdesc Represents a ListDeploymentsResponse.
         * @implements IListDeploymentsResponse
         * @constructor
         * @param {bentoml.IListDeploymentsResponse=} [properties] Properties to set
         */
        function ListDeploymentsResponse(properties) {
            this.deployments = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListDeploymentsResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.ListDeploymentsResponse
         * @instance
         */
        ListDeploymentsResponse.prototype.status = null;

        /**
         * ListDeploymentsResponse deployments.
         * @member {Array.<bentoml.IDeployment>} deployments
         * @memberof bentoml.ListDeploymentsResponse
         * @instance
         */
        ListDeploymentsResponse.prototype.deployments = $util.emptyArray;

        /**
         * Creates a new ListDeploymentsResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {bentoml.IListDeploymentsResponse=} [properties] Properties to set
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse instance
         */
        ListDeploymentsResponse.create = function create(properties) {
            return new ListDeploymentsResponse(properties);
        };

        /**
         * Encodes the specified ListDeploymentsResponse message. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {bentoml.IListDeploymentsResponse} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.deployments != null && message.deployments.length)
                for (let i = 0; i < message.deployments.length; ++i)
                    $root.bentoml.Deployment.encode(message.deployments[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ListDeploymentsResponse message, length delimited. Does not implicitly {@link bentoml.ListDeploymentsResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {bentoml.IListDeploymentsResponse} message ListDeploymentsResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListDeploymentsResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListDeploymentsResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ListDeploymentsResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListDeploymentsResponse} ListDeploymentsResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListDeploymentsResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListDeploymentsResponse message.
         * @function verify
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ListDeploymentsResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.deployments != null && message.hasOwnProperty("deployments")) {
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
         * @function fromObject
         * @memberof bentoml.ListDeploymentsResponse
         * @static
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
         * Creates a plain object from a ListDeploymentsResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ListDeploymentsResponse
         * @static
         * @param {bentoml.ListDeploymentsResponse} message ListDeploymentsResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ListDeploymentsResponse to JSON.
         * @function toJSON
         * @memberof bentoml.ListDeploymentsResponse
         * @instance
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
         * @memberof bentoml
         * @interface IStatus
         * @property {bentoml.Status.Code|null} [status_code] Status status_code
         * @property {string|null} [error_message] Status error_message
         */

        /**
         * Constructs a new Status.
         * @memberof bentoml
         * @classdesc Represents a Status.
         * @implements IStatus
         * @constructor
         * @param {bentoml.IStatus=} [properties] Properties to set
         */
        function Status(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Status status_code.
         * @member {bentoml.Status.Code} status_code
         * @memberof bentoml.Status
         * @instance
         */
        Status.prototype.status_code = 0;

        /**
         * Status error_message.
         * @member {string} error_message
         * @memberof bentoml.Status
         * @instance
         */
        Status.prototype.error_message = "";

        /**
         * Creates a new Status instance using the specified properties.
         * @function create
         * @memberof bentoml.Status
         * @static
         * @param {bentoml.IStatus=} [properties] Properties to set
         * @returns {bentoml.Status} Status instance
         */
        Status.create = function create(properties) {
            return new Status(properties);
        };

        /**
         * Encodes the specified Status message. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @function encode
         * @memberof bentoml.Status
         * @static
         * @param {bentoml.IStatus} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Status.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status_code != null && Object.hasOwnProperty.call(message, "status_code"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.status_code);
            if (message.error_message != null && Object.hasOwnProperty.call(message, "error_message"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.error_message);
            return writer;
        };

        /**
         * Encodes the specified Status message, length delimited. Does not implicitly {@link bentoml.Status.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.Status
         * @static
         * @param {bentoml.IStatus} message Status message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Status.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Status message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.Status
         * @static
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
                    message.status_code = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.Status
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Status} Status
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Status.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Status message.
         * @function verify
         * @memberof bentoml.Status
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Status.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status_code != null && message.hasOwnProperty("status_code"))
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
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            return null;
        };

        /**
         * Creates a Status message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.Status
         * @static
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
         * Creates a plain object from a Status message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.Status
         * @static
         * @param {bentoml.Status} message Status
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this Status to JSON.
         * @function toJSON
         * @memberof bentoml.Status
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        Status.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Code enum.
         * @name bentoml.Status.Code
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
         * @memberof bentoml
         * @interface IBentoUri
         * @property {bentoml.BentoUri.StorageType|null} [type] BentoUri type
         * @property {string|null} [uri] BentoUri uri
         * @property {string|null} [s3_presigned_url] BentoUri s3_presigned_url
         */

        /**
         * Constructs a new BentoUri.
         * @memberof bentoml
         * @classdesc Represents a BentoUri.
         * @implements IBentoUri
         * @constructor
         * @param {bentoml.IBentoUri=} [properties] Properties to set
         */
        function BentoUri(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * BentoUri type.
         * @member {bentoml.BentoUri.StorageType} type
         * @memberof bentoml.BentoUri
         * @instance
         */
        BentoUri.prototype.type = 0;

        /**
         * BentoUri uri.
         * @member {string} uri
         * @memberof bentoml.BentoUri
         * @instance
         */
        BentoUri.prototype.uri = "";

        /**
         * BentoUri s3_presigned_url.
         * @member {string} s3_presigned_url
         * @memberof bentoml.BentoUri
         * @instance
         */
        BentoUri.prototype.s3_presigned_url = "";

        /**
         * Creates a new BentoUri instance using the specified properties.
         * @function create
         * @memberof bentoml.BentoUri
         * @static
         * @param {bentoml.IBentoUri=} [properties] Properties to set
         * @returns {bentoml.BentoUri} BentoUri instance
         */
        BentoUri.create = function create(properties) {
            return new BentoUri(properties);
        };

        /**
         * Encodes the specified BentoUri message. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @function encode
         * @memberof bentoml.BentoUri
         * @static
         * @param {bentoml.IBentoUri} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoUri.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.type != null && Object.hasOwnProperty.call(message, "type"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.type);
            if (message.uri != null && Object.hasOwnProperty.call(message, "uri"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.uri);
            if (message.s3_presigned_url != null && Object.hasOwnProperty.call(message, "s3_presigned_url"))
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.s3_presigned_url);
            return writer;
        };

        /**
         * Encodes the specified BentoUri message, length delimited. Does not implicitly {@link bentoml.BentoUri.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.BentoUri
         * @static
         * @param {bentoml.IBentoUri} message BentoUri message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoUri.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a BentoUri message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.BentoUri
         * @static
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
                    message.type = reader.int32();
                    break;
                case 2:
                    message.uri = reader.string();
                    break;
                case 3:
                    message.s3_presigned_url = reader.string();
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
         * @function decodeDelimited
         * @memberof bentoml.BentoUri
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoUri} BentoUri
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoUri.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a BentoUri message.
         * @function verify
         * @memberof bentoml.BentoUri
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        BentoUri.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.type != null && message.hasOwnProperty("type"))
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
            if (message.uri != null && message.hasOwnProperty("uri"))
                if (!$util.isString(message.uri))
                    return "uri: string expected";
            if (message.s3_presigned_url != null && message.hasOwnProperty("s3_presigned_url"))
                if (!$util.isString(message.s3_presigned_url))
                    return "s3_presigned_url: string expected";
            return null;
        };

        /**
         * Creates a BentoUri message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.BentoUri
         * @static
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
            if (object.s3_presigned_url != null)
                message.s3_presigned_url = String(object.s3_presigned_url);
            return message;
        };

        /**
         * Creates a plain object from a BentoUri message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.BentoUri
         * @static
         * @param {bentoml.BentoUri} message BentoUri
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        BentoUri.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.type = options.enums === String ? "UNSET" : 0;
                object.uri = "";
                object.s3_presigned_url = "";
            }
            if (message.type != null && message.hasOwnProperty("type"))
                object.type = options.enums === String ? $root.bentoml.BentoUri.StorageType[message.type] : message.type;
            if (message.uri != null && message.hasOwnProperty("uri"))
                object.uri = message.uri;
            if (message.s3_presigned_url != null && message.hasOwnProperty("s3_presigned_url"))
                object.s3_presigned_url = message.s3_presigned_url;
            return object;
        };

        /**
         * Converts this BentoUri to JSON.
         * @function toJSON
         * @memberof bentoml.BentoUri
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        BentoUri.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * StorageType enum.
         * @name bentoml.BentoUri.StorageType
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
         * @memberof bentoml
         * @interface IBentoServiceMetadata
         * @property {string|null} [name] BentoServiceMetadata name
         * @property {string|null} [version] BentoServiceMetadata version
         * @property {google.protobuf.ITimestamp|null} [created_at] BentoServiceMetadata created_at
         * @property {bentoml.BentoServiceMetadata.IBentoServiceEnv|null} [env] BentoServiceMetadata env
         * @property {Array.<bentoml.BentoServiceMetadata.IBentoArtifact>|null} [artifacts] BentoServiceMetadata artifacts
         * @property {Array.<bentoml.BentoServiceMetadata.IBentoServiceApi>|null} [apis] BentoServiceMetadata apis
         */

        /**
         * Constructs a new BentoServiceMetadata.
         * @memberof bentoml
         * @classdesc Represents a BentoServiceMetadata.
         * @implements IBentoServiceMetadata
         * @constructor
         * @param {bentoml.IBentoServiceMetadata=} [properties] Properties to set
         */
        function BentoServiceMetadata(properties) {
            this.artifacts = [];
            this.apis = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * BentoServiceMetadata name.
         * @member {string} name
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.name = "";

        /**
         * BentoServiceMetadata version.
         * @member {string} version
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.version = "";

        /**
         * BentoServiceMetadata created_at.
         * @member {google.protobuf.ITimestamp|null|undefined} created_at
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.created_at = null;

        /**
         * BentoServiceMetadata env.
         * @member {bentoml.BentoServiceMetadata.IBentoServiceEnv|null|undefined} env
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.env = null;

        /**
         * BentoServiceMetadata artifacts.
         * @member {Array.<bentoml.BentoServiceMetadata.IBentoArtifact>} artifacts
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.artifacts = $util.emptyArray;

        /**
         * BentoServiceMetadata apis.
         * @member {Array.<bentoml.BentoServiceMetadata.IBentoServiceApi>} apis
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         */
        BentoServiceMetadata.prototype.apis = $util.emptyArray;

        /**
         * Creates a new BentoServiceMetadata instance using the specified properties.
         * @function create
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {bentoml.IBentoServiceMetadata=} [properties] Properties to set
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata instance
         */
        BentoServiceMetadata.create = function create(properties) {
            return new BentoServiceMetadata(properties);
        };

        /**
         * Encodes the specified BentoServiceMetadata message. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @function encode
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {bentoml.IBentoServiceMetadata} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoServiceMetadata.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
            if (message.version != null && Object.hasOwnProperty.call(message, "version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            if (message.created_at != null && Object.hasOwnProperty.call(message, "created_at"))
                $root.google.protobuf.Timestamp.encode(message.created_at, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.env != null && Object.hasOwnProperty.call(message, "env"))
                $root.bentoml.BentoServiceMetadata.BentoServiceEnv.encode(message.env, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.artifacts != null && message.artifacts.length)
                for (let i = 0; i < message.artifacts.length; ++i)
                    $root.bentoml.BentoServiceMetadata.BentoArtifact.encode(message.artifacts[i], writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            if (message.apis != null && message.apis.length)
                for (let i = 0; i < message.apis.length; ++i)
                    $root.bentoml.BentoServiceMetadata.BentoServiceApi.encode(message.apis[i], writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified BentoServiceMetadata message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {bentoml.IBentoServiceMetadata} message BentoServiceMetadata message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        BentoServiceMetadata.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a BentoServiceMetadata message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.BentoServiceMetadata
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.BentoServiceMetadata} BentoServiceMetadata
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        BentoServiceMetadata.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a BentoServiceMetadata message.
         * @function verify
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        BentoServiceMetadata.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.name != null && message.hasOwnProperty("name"))
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.version != null && message.hasOwnProperty("version"))
                if (!$util.isString(message.version))
                    return "version: string expected";
            if (message.created_at != null && message.hasOwnProperty("created_at")) {
                let error = $root.google.protobuf.Timestamp.verify(message.created_at);
                if (error)
                    return "created_at." + error;
            }
            if (message.env != null && message.hasOwnProperty("env")) {
                let error = $root.bentoml.BentoServiceMetadata.BentoServiceEnv.verify(message.env);
                if (error)
                    return "env." + error;
            }
            if (message.artifacts != null && message.hasOwnProperty("artifacts")) {
                if (!Array.isArray(message.artifacts))
                    return "artifacts: array expected";
                for (let i = 0; i < message.artifacts.length; ++i) {
                    let error = $root.bentoml.BentoServiceMetadata.BentoArtifact.verify(message.artifacts[i]);
                    if (error)
                        return "artifacts." + error;
                }
            }
            if (message.apis != null && message.hasOwnProperty("apis")) {
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
         * @function fromObject
         * @memberof bentoml.BentoServiceMetadata
         * @static
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
         * Creates a plain object from a BentoServiceMetadata message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.BentoServiceMetadata
         * @static
         * @param {bentoml.BentoServiceMetadata} message BentoServiceMetadata
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this BentoServiceMetadata to JSON.
         * @function toJSON
         * @memberof bentoml.BentoServiceMetadata
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        BentoServiceMetadata.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        BentoServiceMetadata.BentoServiceEnv = (function() {

            /**
             * Properties of a BentoServiceEnv.
             * @memberof bentoml.BentoServiceMetadata
             * @interface IBentoServiceEnv
             * @property {string|null} [setup_sh] BentoServiceEnv setup_sh
             * @property {string|null} [conda_env] BentoServiceEnv conda_env
             * @property {string|null} [pip_dependencies] BentoServiceEnv pip_dependencies
             * @property {string|null} [python_version] BentoServiceEnv python_version
             * @property {string|null} [docker_base_image] BentoServiceEnv docker_base_image
             */

            /**
             * Constructs a new BentoServiceEnv.
             * @memberof bentoml.BentoServiceMetadata
             * @classdesc Represents a BentoServiceEnv.
             * @implements IBentoServiceEnv
             * @constructor
             * @param {bentoml.BentoServiceMetadata.IBentoServiceEnv=} [properties] Properties to set
             */
            function BentoServiceEnv(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoServiceEnv setup_sh.
             * @member {string} setup_sh
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
             */
            BentoServiceEnv.prototype.setup_sh = "";

            /**
             * BentoServiceEnv conda_env.
             * @member {string} conda_env
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
             */
            BentoServiceEnv.prototype.conda_env = "";

            /**
             * BentoServiceEnv pip_dependencies.
             * @member {string} pip_dependencies
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
             */
            BentoServiceEnv.prototype.pip_dependencies = "";

            /**
             * BentoServiceEnv python_version.
             * @member {string} python_version
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
             */
            BentoServiceEnv.prototype.python_version = "";

            /**
             * BentoServiceEnv docker_base_image.
             * @member {string} docker_base_image
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
             */
            BentoServiceEnv.prototype.docker_base_image = "";

            /**
             * Creates a new BentoServiceEnv instance using the specified properties.
             * @function create
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceEnv=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv instance
             */
            BentoServiceEnv.create = function create(properties) {
                return new BentoServiceEnv(properties);
            };

            /**
             * Encodes the specified BentoServiceEnv message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @function encode
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceEnv} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceEnv.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.setup_sh != null && Object.hasOwnProperty.call(message, "setup_sh"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.setup_sh);
                if (message.conda_env != null && Object.hasOwnProperty.call(message, "conda_env"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.conda_env);
                if (message.pip_dependencies != null && Object.hasOwnProperty.call(message, "pip_dependencies"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.pip_dependencies);
                if (message.python_version != null && Object.hasOwnProperty.call(message, "python_version"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.python_version);
                if (message.docker_base_image != null && Object.hasOwnProperty.call(message, "docker_base_image"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.docker_base_image);
                return writer;
            };

            /**
             * Encodes the specified BentoServiceEnv message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceEnv.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceEnv} message BentoServiceEnv message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceEnv.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoServiceEnv message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceEnv} BentoServiceEnv
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceEnv.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoServiceEnv message.
             * @function verify
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            BentoServiceEnv.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.setup_sh != null && message.hasOwnProperty("setup_sh"))
                    if (!$util.isString(message.setup_sh))
                        return "setup_sh: string expected";
                if (message.conda_env != null && message.hasOwnProperty("conda_env"))
                    if (!$util.isString(message.conda_env))
                        return "conda_env: string expected";
                if (message.pip_dependencies != null && message.hasOwnProperty("pip_dependencies"))
                    if (!$util.isString(message.pip_dependencies))
                        return "pip_dependencies: string expected";
                if (message.python_version != null && message.hasOwnProperty("python_version"))
                    if (!$util.isString(message.python_version))
                        return "python_version: string expected";
                if (message.docker_base_image != null && message.hasOwnProperty("docker_base_image"))
                    if (!$util.isString(message.docker_base_image))
                        return "docker_base_image: string expected";
                return null;
            };

            /**
             * Creates a BentoServiceEnv message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
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
             * Creates a plain object from a BentoServiceEnv message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @static
             * @param {bentoml.BentoServiceMetadata.BentoServiceEnv} message BentoServiceEnv
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this BentoServiceEnv to JSON.
             * @function toJSON
             * @memberof bentoml.BentoServiceMetadata.BentoServiceEnv
             * @instance
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
             * @memberof bentoml.BentoServiceMetadata
             * @interface IBentoArtifact
             * @property {string|null} [name] BentoArtifact name
             * @property {string|null} [artifact_type] BentoArtifact artifact_type
             */

            /**
             * Constructs a new BentoArtifact.
             * @memberof bentoml.BentoServiceMetadata
             * @classdesc Represents a BentoArtifact.
             * @implements IBentoArtifact
             * @constructor
             * @param {bentoml.BentoServiceMetadata.IBentoArtifact=} [properties] Properties to set
             */
            function BentoArtifact(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoArtifact name.
             * @member {string} name
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @instance
             */
            BentoArtifact.prototype.name = "";

            /**
             * BentoArtifact artifact_type.
             * @member {string} artifact_type
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @instance
             */
            BentoArtifact.prototype.artifact_type = "";

            /**
             * Creates a new BentoArtifact instance using the specified properties.
             * @function create
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoArtifact=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact instance
             */
            BentoArtifact.create = function create(properties) {
                return new BentoArtifact(properties);
            };

            /**
             * Encodes the specified BentoArtifact message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @function encode
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoArtifact} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoArtifact.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.artifact_type != null && Object.hasOwnProperty.call(message, "artifact_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.artifact_type);
                return writer;
            };

            /**
             * Encodes the specified BentoArtifact message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoArtifact.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoArtifact} message BentoArtifact message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoArtifact.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoArtifact message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoArtifact} BentoArtifact
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoArtifact.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoArtifact message.
             * @function verify
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            BentoArtifact.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.artifact_type != null && message.hasOwnProperty("artifact_type"))
                    if (!$util.isString(message.artifact_type))
                        return "artifact_type: string expected";
                return null;
            };

            /**
             * Creates a BentoArtifact message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
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
             * Creates a plain object from a BentoArtifact message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @static
             * @param {bentoml.BentoServiceMetadata.BentoArtifact} message BentoArtifact
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this BentoArtifact to JSON.
             * @function toJSON
             * @memberof bentoml.BentoServiceMetadata.BentoArtifact
             * @instance
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
             * @memberof bentoml.BentoServiceMetadata
             * @interface IBentoServiceApi
             * @property {string|null} [name] BentoServiceApi name
             * @property {string|null} [input_type] BentoServiceApi input_type
             * @property {string|null} [docs] BentoServiceApi docs
             * @property {google.protobuf.IStruct|null} [input_config] BentoServiceApi input_config
             * @property {google.protobuf.IStruct|null} [output_config] BentoServiceApi output_config
             * @property {string|null} [output_type] BentoServiceApi output_type
             * @property {number|null} [mb_max_latency] BentoServiceApi mb_max_latency
             * @property {number|null} [mb_max_batch_size] BentoServiceApi mb_max_batch_size
             */

            /**
             * Constructs a new BentoServiceApi.
             * @memberof bentoml.BentoServiceMetadata
             * @classdesc Represents a BentoServiceApi.
             * @implements IBentoServiceApi
             * @constructor
             * @param {bentoml.BentoServiceMetadata.IBentoServiceApi=} [properties] Properties to set
             */
            function BentoServiceApi(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BentoServiceApi name.
             * @member {string} name
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.name = "";

            /**
             * BentoServiceApi input_type.
             * @member {string} input_type
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.input_type = "";

            /**
             * BentoServiceApi docs.
             * @member {string} docs
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.docs = "";

            /**
             * BentoServiceApi input_config.
             * @member {google.protobuf.IStruct|null|undefined} input_config
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.input_config = null;

            /**
             * BentoServiceApi output_config.
             * @member {google.protobuf.IStruct|null|undefined} output_config
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.output_config = null;

            /**
             * BentoServiceApi output_type.
             * @member {string} output_type
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.output_type = "";

            /**
             * BentoServiceApi mb_max_latency.
             * @member {number} mb_max_latency
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.mb_max_latency = 0;

            /**
             * BentoServiceApi mb_max_batch_size.
             * @member {number} mb_max_batch_size
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
             */
            BentoServiceApi.prototype.mb_max_batch_size = 0;

            /**
             * Creates a new BentoServiceApi instance using the specified properties.
             * @function create
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceApi=} [properties] Properties to set
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi instance
             */
            BentoServiceApi.create = function create(properties) {
                return new BentoServiceApi(properties);
            };

            /**
             * Encodes the specified BentoServiceApi message. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @function encode
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceApi} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceApi.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
                if (message.input_type != null && Object.hasOwnProperty.call(message, "input_type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.input_type);
                if (message.docs != null && Object.hasOwnProperty.call(message, "docs"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.docs);
                if (message.input_config != null && Object.hasOwnProperty.call(message, "input_config"))
                    $root.google.protobuf.Struct.encode(message.input_config, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.output_config != null && Object.hasOwnProperty.call(message, "output_config"))
                    $root.google.protobuf.Struct.encode(message.output_config, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.output_type != null && Object.hasOwnProperty.call(message, "output_type"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.output_type);
                if (message.mb_max_latency != null && Object.hasOwnProperty.call(message, "mb_max_latency"))
                    writer.uint32(/* id 7, wireType 0 =*/56).int32(message.mb_max_latency);
                if (message.mb_max_batch_size != null && Object.hasOwnProperty.call(message, "mb_max_batch_size"))
                    writer.uint32(/* id 8, wireType 0 =*/64).int32(message.mb_max_batch_size);
                return writer;
            };

            /**
             * Encodes the specified BentoServiceApi message, length delimited. Does not implicitly {@link bentoml.BentoServiceMetadata.BentoServiceApi.verify|verify} messages.
             * @function encodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {bentoml.BentoServiceMetadata.IBentoServiceApi} message BentoServiceApi message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            BentoServiceApi.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a BentoServiceApi message from the specified reader or buffer.
             * @function decode
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
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
             * @function decodeDelimited
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {bentoml.BentoServiceMetadata.BentoServiceApi} BentoServiceApi
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            BentoServiceApi.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a BentoServiceApi message.
             * @function verify
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            BentoServiceApi.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.input_type != null && message.hasOwnProperty("input_type"))
                    if (!$util.isString(message.input_type))
                        return "input_type: string expected";
                if (message.docs != null && message.hasOwnProperty("docs"))
                    if (!$util.isString(message.docs))
                        return "docs: string expected";
                if (message.input_config != null && message.hasOwnProperty("input_config")) {
                    let error = $root.google.protobuf.Struct.verify(message.input_config);
                    if (error)
                        return "input_config." + error;
                }
                if (message.output_config != null && message.hasOwnProperty("output_config")) {
                    let error = $root.google.protobuf.Struct.verify(message.output_config);
                    if (error)
                        return "output_config." + error;
                }
                if (message.output_type != null && message.hasOwnProperty("output_type"))
                    if (!$util.isString(message.output_type))
                        return "output_type: string expected";
                if (message.mb_max_latency != null && message.hasOwnProperty("mb_max_latency"))
                    if (!$util.isInteger(message.mb_max_latency))
                        return "mb_max_latency: integer expected";
                if (message.mb_max_batch_size != null && message.hasOwnProperty("mb_max_batch_size"))
                    if (!$util.isInteger(message.mb_max_batch_size))
                        return "mb_max_batch_size: integer expected";
                return null;
            };

            /**
             * Creates a BentoServiceApi message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
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
             * Creates a plain object from a BentoServiceApi message. Also converts values to other types if specified.
             * @function toObject
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @static
             * @param {bentoml.BentoServiceMetadata.BentoServiceApi} message BentoServiceApi
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this BentoServiceApi to JSON.
             * @function toJSON
             * @memberof bentoml.BentoServiceMetadata.BentoServiceApi
             * @instance
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
         * @memberof bentoml
         * @interface IBento
         * @property {string|null} [name] Bento name
         * @property {string|null} [version] Bento version
         * @property {bentoml.IBentoUri|null} [uri] Bento uri
         * @property {bentoml.IBentoServiceMetadata|null} [bento_service_metadata] Bento bento_service_metadata
         * @property {bentoml.IUploadStatus|null} [status] Bento status
         */

        /**
         * Constructs a new Bento.
         * @memberof bentoml
         * @classdesc Represents a Bento.
         * @implements IBento
         * @constructor
         * @param {bentoml.IBento=} [properties] Properties to set
         */
        function Bento(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Bento name.
         * @member {string} name
         * @memberof bentoml.Bento
         * @instance
         */
        Bento.prototype.name = "";

        /**
         * Bento version.
         * @member {string} version
         * @memberof bentoml.Bento
         * @instance
         */
        Bento.prototype.version = "";

        /**
         * Bento uri.
         * @member {bentoml.IBentoUri|null|undefined} uri
         * @memberof bentoml.Bento
         * @instance
         */
        Bento.prototype.uri = null;

        /**
         * Bento bento_service_metadata.
         * @member {bentoml.IBentoServiceMetadata|null|undefined} bento_service_metadata
         * @memberof bentoml.Bento
         * @instance
         */
        Bento.prototype.bento_service_metadata = null;

        /**
         * Bento status.
         * @member {bentoml.IUploadStatus|null|undefined} status
         * @memberof bentoml.Bento
         * @instance
         */
        Bento.prototype.status = null;

        /**
         * Creates a new Bento instance using the specified properties.
         * @function create
         * @memberof bentoml.Bento
         * @static
         * @param {bentoml.IBento=} [properties] Properties to set
         * @returns {bentoml.Bento} Bento instance
         */
        Bento.create = function create(properties) {
            return new Bento(properties);
        };

        /**
         * Encodes the specified Bento message. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @function encode
         * @memberof bentoml.Bento
         * @static
         * @param {bentoml.IBento} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Bento.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.name != null && Object.hasOwnProperty.call(message, "name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.name);
            if (message.version != null && Object.hasOwnProperty.call(message, "version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            if (message.uri != null && Object.hasOwnProperty.call(message, "uri"))
                $root.bentoml.BentoUri.encode(message.uri, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.bento_service_metadata != null && Object.hasOwnProperty.call(message, "bento_service_metadata"))
                $root.bentoml.BentoServiceMetadata.encode(message.bento_service_metadata, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.UploadStatus.encode(message.status, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified Bento message, length delimited. Does not implicitly {@link bentoml.Bento.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.Bento
         * @static
         * @param {bentoml.IBento} message Bento message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Bento.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Bento message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.Bento
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.Bento
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Bento} Bento
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Bento.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Bento message.
         * @function verify
         * @memberof bentoml.Bento
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Bento.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.name != null && message.hasOwnProperty("name"))
                if (!$util.isString(message.name))
                    return "name: string expected";
            if (message.version != null && message.hasOwnProperty("version"))
                if (!$util.isString(message.version))
                    return "version: string expected";
            if (message.uri != null && message.hasOwnProperty("uri")) {
                let error = $root.bentoml.BentoUri.verify(message.uri);
                if (error)
                    return "uri." + error;
            }
            if (message.bento_service_metadata != null && message.hasOwnProperty("bento_service_metadata")) {
                let error = $root.bentoml.BentoServiceMetadata.verify(message.bento_service_metadata);
                if (error)
                    return "bento_service_metadata." + error;
            }
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.UploadStatus.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a Bento message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.Bento
         * @static
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
         * Creates a plain object from a Bento message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.Bento
         * @static
         * @param {bentoml.Bento} message Bento
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this Bento to JSON.
         * @function toJSON
         * @memberof bentoml.Bento
         * @instance
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
         * @memberof bentoml
         * @interface IAddBentoRequest
         * @property {string|null} [bento_name] AddBentoRequest bento_name
         * @property {string|null} [bento_version] AddBentoRequest bento_version
         */

        /**
         * Constructs a new AddBentoRequest.
         * @memberof bentoml
         * @classdesc Represents an AddBentoRequest.
         * @implements IAddBentoRequest
         * @constructor
         * @param {bentoml.IAddBentoRequest=} [properties] Properties to set
         */
        function AddBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * AddBentoRequest bento_name.
         * @member {string} bento_name
         * @memberof bentoml.AddBentoRequest
         * @instance
         */
        AddBentoRequest.prototype.bento_name = "";

        /**
         * AddBentoRequest bento_version.
         * @member {string} bento_version
         * @memberof bentoml.AddBentoRequest
         * @instance
         */
        AddBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new AddBentoRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {bentoml.IAddBentoRequest=} [properties] Properties to set
         * @returns {bentoml.AddBentoRequest} AddBentoRequest instance
         */
        AddBentoRequest.create = function create(properties) {
            return new AddBentoRequest(properties);
        };

        /**
         * Encodes the specified AddBentoRequest message. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {bentoml.IAddBentoRequest} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && Object.hasOwnProperty.call(message, "bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified AddBentoRequest message, length delimited. Does not implicitly {@link bentoml.AddBentoRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {bentoml.IAddBentoRequest} message AddBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an AddBentoRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.AddBentoRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoRequest} AddBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an AddBentoRequest message.
         * @function verify
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        AddBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates an AddBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.AddBentoRequest
         * @static
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
         * Creates a plain object from an AddBentoRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.AddBentoRequest
         * @static
         * @param {bentoml.AddBentoRequest} message AddBentoRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this AddBentoRequest to JSON.
         * @function toJSON
         * @memberof bentoml.AddBentoRequest
         * @instance
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
         * @memberof bentoml
         * @interface IAddBentoResponse
         * @property {bentoml.IStatus|null} [status] AddBentoResponse status
         * @property {bentoml.IBentoUri|null} [uri] AddBentoResponse uri
         */

        /**
         * Constructs a new AddBentoResponse.
         * @memberof bentoml
         * @classdesc Represents an AddBentoResponse.
         * @implements IAddBentoResponse
         * @constructor
         * @param {bentoml.IAddBentoResponse=} [properties] Properties to set
         */
        function AddBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * AddBentoResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.AddBentoResponse
         * @instance
         */
        AddBentoResponse.prototype.status = null;

        /**
         * AddBentoResponse uri.
         * @member {bentoml.IBentoUri|null|undefined} uri
         * @memberof bentoml.AddBentoResponse
         * @instance
         */
        AddBentoResponse.prototype.uri = null;

        /**
         * Creates a new AddBentoResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {bentoml.IAddBentoResponse=} [properties] Properties to set
         * @returns {bentoml.AddBentoResponse} AddBentoResponse instance
         */
        AddBentoResponse.create = function create(properties) {
            return new AddBentoResponse(properties);
        };

        /**
         * Encodes the specified AddBentoResponse message. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {bentoml.IAddBentoResponse} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.uri != null && Object.hasOwnProperty.call(message, "uri"))
                $root.bentoml.BentoUri.encode(message.uri, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified AddBentoResponse message, length delimited. Does not implicitly {@link bentoml.AddBentoResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {bentoml.IAddBentoResponse} message AddBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        AddBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an AddBentoResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.AddBentoResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.AddBentoResponse} AddBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        AddBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an AddBentoResponse message.
         * @function verify
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        AddBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.uri != null && message.hasOwnProperty("uri")) {
                let error = $root.bentoml.BentoUri.verify(message.uri);
                if (error)
                    return "uri." + error;
            }
            return null;
        };

        /**
         * Creates an AddBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.AddBentoResponse
         * @static
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
         * Creates a plain object from an AddBentoResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.AddBentoResponse
         * @static
         * @param {bentoml.AddBentoResponse} message AddBentoResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this AddBentoResponse to JSON.
         * @function toJSON
         * @memberof bentoml.AddBentoResponse
         * @instance
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
         * @memberof bentoml
         * @interface IUploadStatus
         * @property {bentoml.UploadStatus.Status|null} [status] UploadStatus status
         * @property {google.protobuf.ITimestamp|null} [updated_at] UploadStatus updated_at
         * @property {number|null} [percentage] UploadStatus percentage
         * @property {string|null} [error_message] UploadStatus error_message
         */

        /**
         * Constructs a new UploadStatus.
         * @memberof bentoml
         * @classdesc Represents an UploadStatus.
         * @implements IUploadStatus
         * @constructor
         * @param {bentoml.IUploadStatus=} [properties] Properties to set
         */
        function UploadStatus(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * UploadStatus status.
         * @member {bentoml.UploadStatus.Status} status
         * @memberof bentoml.UploadStatus
         * @instance
         */
        UploadStatus.prototype.status = 0;

        /**
         * UploadStatus updated_at.
         * @member {google.protobuf.ITimestamp|null|undefined} updated_at
         * @memberof bentoml.UploadStatus
         * @instance
         */
        UploadStatus.prototype.updated_at = null;

        /**
         * UploadStatus percentage.
         * @member {number} percentage
         * @memberof bentoml.UploadStatus
         * @instance
         */
        UploadStatus.prototype.percentage = 0;

        /**
         * UploadStatus error_message.
         * @member {string} error_message
         * @memberof bentoml.UploadStatus
         * @instance
         */
        UploadStatus.prototype.error_message = "";

        /**
         * Creates a new UploadStatus instance using the specified properties.
         * @function create
         * @memberof bentoml.UploadStatus
         * @static
         * @param {bentoml.IUploadStatus=} [properties] Properties to set
         * @returns {bentoml.UploadStatus} UploadStatus instance
         */
        UploadStatus.create = function create(properties) {
            return new UploadStatus(properties);
        };

        /**
         * Encodes the specified UploadStatus message. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @function encode
         * @memberof bentoml.UploadStatus
         * @static
         * @param {bentoml.IUploadStatus} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UploadStatus.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.status);
            if (message.updated_at != null && Object.hasOwnProperty.call(message, "updated_at"))
                $root.google.protobuf.Timestamp.encode(message.updated_at, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            if (message.percentage != null && Object.hasOwnProperty.call(message, "percentage"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.percentage);
            if (message.error_message != null && Object.hasOwnProperty.call(message, "error_message"))
                writer.uint32(/* id 4, wireType 2 =*/34).string(message.error_message);
            return writer;
        };

        /**
         * Encodes the specified UploadStatus message, length delimited. Does not implicitly {@link bentoml.UploadStatus.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.UploadStatus
         * @static
         * @param {bentoml.IUploadStatus} message UploadStatus message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UploadStatus.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UploadStatus message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.UploadStatus
         * @static
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
                    message.status = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.UploadStatus
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UploadStatus} UploadStatus
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UploadStatus.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UploadStatus message.
         * @function verify
         * @memberof bentoml.UploadStatus
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        UploadStatus.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status"))
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
            if (message.updated_at != null && message.hasOwnProperty("updated_at")) {
                let error = $root.google.protobuf.Timestamp.verify(message.updated_at);
                if (error)
                    return "updated_at." + error;
            }
            if (message.percentage != null && message.hasOwnProperty("percentage"))
                if (!$util.isInteger(message.percentage))
                    return "percentage: integer expected";
            if (message.error_message != null && message.hasOwnProperty("error_message"))
                if (!$util.isString(message.error_message))
                    return "error_message: string expected";
            return null;
        };

        /**
         * Creates an UploadStatus message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.UploadStatus
         * @static
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
         * Creates a plain object from an UploadStatus message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.UploadStatus
         * @static
         * @param {bentoml.UploadStatus} message UploadStatus
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this UploadStatus to JSON.
         * @function toJSON
         * @memberof bentoml.UploadStatus
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        UploadStatus.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Status enum.
         * @name bentoml.UploadStatus.Status
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
         * @memberof bentoml
         * @interface IUpdateBentoRequest
         * @property {string|null} [bento_name] UpdateBentoRequest bento_name
         * @property {string|null} [bento_version] UpdateBentoRequest bento_version
         * @property {bentoml.IUploadStatus|null} [upload_status] UpdateBentoRequest upload_status
         * @property {bentoml.IBentoServiceMetadata|null} [service_metadata] UpdateBentoRequest service_metadata
         */

        /**
         * Constructs a new UpdateBentoRequest.
         * @memberof bentoml
         * @classdesc Represents an UpdateBentoRequest.
         * @implements IUpdateBentoRequest
         * @constructor
         * @param {bentoml.IUpdateBentoRequest=} [properties] Properties to set
         */
        function UpdateBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * UpdateBentoRequest bento_name.
         * @member {string} bento_name
         * @memberof bentoml.UpdateBentoRequest
         * @instance
         */
        UpdateBentoRequest.prototype.bento_name = "";

        /**
         * UpdateBentoRequest bento_version.
         * @member {string} bento_version
         * @memberof bentoml.UpdateBentoRequest
         * @instance
         */
        UpdateBentoRequest.prototype.bento_version = "";

        /**
         * UpdateBentoRequest upload_status.
         * @member {bentoml.IUploadStatus|null|undefined} upload_status
         * @memberof bentoml.UpdateBentoRequest
         * @instance
         */
        UpdateBentoRequest.prototype.upload_status = null;

        /**
         * UpdateBentoRequest service_metadata.
         * @member {bentoml.IBentoServiceMetadata|null|undefined} service_metadata
         * @memberof bentoml.UpdateBentoRequest
         * @instance
         */
        UpdateBentoRequest.prototype.service_metadata = null;

        /**
         * Creates a new UpdateBentoRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {bentoml.IUpdateBentoRequest=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest instance
         */
        UpdateBentoRequest.create = function create(properties) {
            return new UpdateBentoRequest(properties);
        };

        /**
         * Encodes the specified UpdateBentoRequest message. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {bentoml.IUpdateBentoRequest} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && Object.hasOwnProperty.call(message, "bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            if (message.upload_status != null && Object.hasOwnProperty.call(message, "upload_status"))
                $root.bentoml.UploadStatus.encode(message.upload_status, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
            if (message.service_metadata != null && Object.hasOwnProperty.call(message, "service_metadata"))
                $root.bentoml.BentoServiceMetadata.encode(message.service_metadata, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified UpdateBentoRequest message, length delimited. Does not implicitly {@link bentoml.UpdateBentoRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {bentoml.IUpdateBentoRequest} message UpdateBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UpdateBentoRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.UpdateBentoRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoRequest} UpdateBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UpdateBentoRequest message.
         * @function verify
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        UpdateBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            if (message.upload_status != null && message.hasOwnProperty("upload_status")) {
                let error = $root.bentoml.UploadStatus.verify(message.upload_status);
                if (error)
                    return "upload_status." + error;
            }
            if (message.service_metadata != null && message.hasOwnProperty("service_metadata")) {
                let error = $root.bentoml.BentoServiceMetadata.verify(message.service_metadata);
                if (error)
                    return "service_metadata." + error;
            }
            return null;
        };

        /**
         * Creates an UpdateBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.UpdateBentoRequest
         * @static
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
         * Creates a plain object from an UpdateBentoRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.UpdateBentoRequest
         * @static
         * @param {bentoml.UpdateBentoRequest} message UpdateBentoRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this UpdateBentoRequest to JSON.
         * @function toJSON
         * @memberof bentoml.UpdateBentoRequest
         * @instance
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
         * @memberof bentoml
         * @interface IUpdateBentoResponse
         * @property {bentoml.IStatus|null} [status] UpdateBentoResponse status
         */

        /**
         * Constructs a new UpdateBentoResponse.
         * @memberof bentoml
         * @classdesc Represents an UpdateBentoResponse.
         * @implements IUpdateBentoResponse
         * @constructor
         * @param {bentoml.IUpdateBentoResponse=} [properties] Properties to set
         */
        function UpdateBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * UpdateBentoResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.UpdateBentoResponse
         * @instance
         */
        UpdateBentoResponse.prototype.status = null;

        /**
         * Creates a new UpdateBentoResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {bentoml.IUpdateBentoResponse=} [properties] Properties to set
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse instance
         */
        UpdateBentoResponse.create = function create(properties) {
            return new UpdateBentoResponse(properties);
        };

        /**
         * Encodes the specified UpdateBentoResponse message. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {bentoml.IUpdateBentoResponse} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified UpdateBentoResponse message, length delimited. Does not implicitly {@link bentoml.UpdateBentoResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {bentoml.IUpdateBentoResponse} message UpdateBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        UpdateBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes an UpdateBentoResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.UpdateBentoResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.UpdateBentoResponse} UpdateBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        UpdateBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies an UpdateBentoResponse message.
         * @function verify
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        UpdateBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates an UpdateBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.UpdateBentoResponse
         * @static
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
         * Creates a plain object from an UpdateBentoResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.UpdateBentoResponse
         * @static
         * @param {bentoml.UpdateBentoResponse} message UpdateBentoResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this UpdateBentoResponse to JSON.
         * @function toJSON
         * @memberof bentoml.UpdateBentoResponse
         * @instance
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
         * @memberof bentoml
         * @interface IDangerouslyDeleteBentoRequest
         * @property {string|null} [bento_name] DangerouslyDeleteBentoRequest bento_name
         * @property {string|null} [bento_version] DangerouslyDeleteBentoRequest bento_version
         */

        /**
         * Constructs a new DangerouslyDeleteBentoRequest.
         * @memberof bentoml
         * @classdesc Represents a DangerouslyDeleteBentoRequest.
         * @implements IDangerouslyDeleteBentoRequest
         * @constructor
         * @param {bentoml.IDangerouslyDeleteBentoRequest=} [properties] Properties to set
         */
        function DangerouslyDeleteBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DangerouslyDeleteBentoRequest bento_name.
         * @member {string} bento_name
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @instance
         */
        DangerouslyDeleteBentoRequest.prototype.bento_name = "";

        /**
         * DangerouslyDeleteBentoRequest bento_version.
         * @member {string} bento_version
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @instance
         */
        DangerouslyDeleteBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new DangerouslyDeleteBentoRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoRequest=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest instance
         */
        DangerouslyDeleteBentoRequest.create = function create(properties) {
            return new DangerouslyDeleteBentoRequest(properties);
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoRequest} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && Object.hasOwnProperty.call(message, "bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoRequest message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoRequest} message DangerouslyDeleteBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DangerouslyDeleteBentoRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoRequest} DangerouslyDeleteBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DangerouslyDeleteBentoRequest message.
         * @function verify
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DangerouslyDeleteBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates a DangerouslyDeleteBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
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
         * Creates a plain object from a DangerouslyDeleteBentoRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @static
         * @param {bentoml.DangerouslyDeleteBentoRequest} message DangerouslyDeleteBentoRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DangerouslyDeleteBentoRequest to JSON.
         * @function toJSON
         * @memberof bentoml.DangerouslyDeleteBentoRequest
         * @instance
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
         * @memberof bentoml
         * @interface IDangerouslyDeleteBentoResponse
         * @property {bentoml.IStatus|null} [status] DangerouslyDeleteBentoResponse status
         */

        /**
         * Constructs a new DangerouslyDeleteBentoResponse.
         * @memberof bentoml
         * @classdesc Represents a DangerouslyDeleteBentoResponse.
         * @implements IDangerouslyDeleteBentoResponse
         * @constructor
         * @param {bentoml.IDangerouslyDeleteBentoResponse=} [properties] Properties to set
         */
        function DangerouslyDeleteBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * DangerouslyDeleteBentoResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @instance
         */
        DangerouslyDeleteBentoResponse.prototype.status = null;

        /**
         * Creates a new DangerouslyDeleteBentoResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoResponse=} [properties] Properties to set
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse instance
         */
        DangerouslyDeleteBentoResponse.create = function create(properties) {
            return new DangerouslyDeleteBentoResponse(properties);
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoResponse} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified DangerouslyDeleteBentoResponse message, length delimited. Does not implicitly {@link bentoml.DangerouslyDeleteBentoResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {bentoml.IDangerouslyDeleteBentoResponse} message DangerouslyDeleteBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        DangerouslyDeleteBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a DangerouslyDeleteBentoResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.DangerouslyDeleteBentoResponse} DangerouslyDeleteBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        DangerouslyDeleteBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a DangerouslyDeleteBentoResponse message.
         * @function verify
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        DangerouslyDeleteBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a DangerouslyDeleteBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
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
         * Creates a plain object from a DangerouslyDeleteBentoResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @static
         * @param {bentoml.DangerouslyDeleteBentoResponse} message DangerouslyDeleteBentoResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this DangerouslyDeleteBentoResponse to JSON.
         * @function toJSON
         * @memberof bentoml.DangerouslyDeleteBentoResponse
         * @instance
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
         * @memberof bentoml
         * @interface IGetBentoRequest
         * @property {string|null} [bento_name] GetBentoRequest bento_name
         * @property {string|null} [bento_version] GetBentoRequest bento_version
         */

        /**
         * Constructs a new GetBentoRequest.
         * @memberof bentoml
         * @classdesc Represents a GetBentoRequest.
         * @implements IGetBentoRequest
         * @constructor
         * @param {bentoml.IGetBentoRequest=} [properties] Properties to set
         */
        function GetBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetBentoRequest bento_name.
         * @member {string} bento_name
         * @memberof bentoml.GetBentoRequest
         * @instance
         */
        GetBentoRequest.prototype.bento_name = "";

        /**
         * GetBentoRequest bento_version.
         * @member {string} bento_version
         * @memberof bentoml.GetBentoRequest
         * @instance
         */
        GetBentoRequest.prototype.bento_version = "";

        /**
         * Creates a new GetBentoRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {bentoml.IGetBentoRequest=} [properties] Properties to set
         * @returns {bentoml.GetBentoRequest} GetBentoRequest instance
         */
        GetBentoRequest.create = function create(properties) {
            return new GetBentoRequest(properties);
        };

        /**
         * Encodes the specified GetBentoRequest message. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {bentoml.IGetBentoRequest} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.bento_version != null && Object.hasOwnProperty.call(message, "bento_version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.bento_version);
            return writer;
        };

        /**
         * Encodes the specified GetBentoRequest message, length delimited. Does not implicitly {@link bentoml.GetBentoRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {bentoml.IGetBentoRequest} message GetBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetBentoRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.GetBentoRequest
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoRequest} GetBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetBentoRequest message.
         * @function verify
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GetBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.bento_version != null && message.hasOwnProperty("bento_version"))
                if (!$util.isString(message.bento_version))
                    return "bento_version: string expected";
            return null;
        };

        /**
         * Creates a GetBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.GetBentoRequest
         * @static
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
         * Creates a plain object from a GetBentoRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.GetBentoRequest
         * @static
         * @param {bentoml.GetBentoRequest} message GetBentoRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this GetBentoRequest to JSON.
         * @function toJSON
         * @memberof bentoml.GetBentoRequest
         * @instance
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
         * @memberof bentoml
         * @interface IGetBentoResponse
         * @property {bentoml.IStatus|null} [status] GetBentoResponse status
         * @property {bentoml.IBento|null} [bento] GetBentoResponse bento
         */

        /**
         * Constructs a new GetBentoResponse.
         * @memberof bentoml
         * @classdesc Represents a GetBentoResponse.
         * @implements IGetBentoResponse
         * @constructor
         * @param {bentoml.IGetBentoResponse=} [properties] Properties to set
         */
        function GetBentoResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetBentoResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.GetBentoResponse
         * @instance
         */
        GetBentoResponse.prototype.status = null;

        /**
         * GetBentoResponse bento.
         * @member {bentoml.IBento|null|undefined} bento
         * @memberof bentoml.GetBentoResponse
         * @instance
         */
        GetBentoResponse.prototype.bento = null;

        /**
         * Creates a new GetBentoResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {bentoml.IGetBentoResponse=} [properties] Properties to set
         * @returns {bentoml.GetBentoResponse} GetBentoResponse instance
         */
        GetBentoResponse.create = function create(properties) {
            return new GetBentoResponse(properties);
        };

        /**
         * Encodes the specified GetBentoResponse message. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {bentoml.IGetBentoResponse} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.bento != null && Object.hasOwnProperty.call(message, "bento"))
                $root.bentoml.Bento.encode(message.bento, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified GetBentoResponse message, length delimited. Does not implicitly {@link bentoml.GetBentoResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {bentoml.IGetBentoResponse} message GetBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetBentoResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.GetBentoResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetBentoResponse} GetBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetBentoResponse message.
         * @function verify
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GetBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.bento != null && message.hasOwnProperty("bento")) {
                let error = $root.bentoml.Bento.verify(message.bento);
                if (error)
                    return "bento." + error;
            }
            return null;
        };

        /**
         * Creates a GetBentoResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.GetBentoResponse
         * @static
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
         * Creates a plain object from a GetBentoResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.GetBentoResponse
         * @static
         * @param {bentoml.GetBentoResponse} message GetBentoResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this GetBentoResponse to JSON.
         * @function toJSON
         * @memberof bentoml.GetBentoResponse
         * @instance
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
         * @memberof bentoml
         * @interface IListBentoRequest
         * @property {string|null} [bento_name] ListBentoRequest bento_name
         * @property {number|null} [offset] ListBentoRequest offset
         * @property {number|null} [limit] ListBentoRequest limit
         * @property {bentoml.ListBentoRequest.SORTABLE_COLUMN|null} [order_by] ListBentoRequest order_by
         * @property {boolean|null} [ascending_order] ListBentoRequest ascending_order
         */

        /**
         * Constructs a new ListBentoRequest.
         * @memberof bentoml
         * @classdesc Represents a ListBentoRequest.
         * @implements IListBentoRequest
         * @constructor
         * @param {bentoml.IListBentoRequest=} [properties] Properties to set
         */
        function ListBentoRequest(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListBentoRequest bento_name.
         * @member {string} bento_name
         * @memberof bentoml.ListBentoRequest
         * @instance
         */
        ListBentoRequest.prototype.bento_name = "";

        /**
         * ListBentoRequest offset.
         * @member {number} offset
         * @memberof bentoml.ListBentoRequest
         * @instance
         */
        ListBentoRequest.prototype.offset = 0;

        /**
         * ListBentoRequest limit.
         * @member {number} limit
         * @memberof bentoml.ListBentoRequest
         * @instance
         */
        ListBentoRequest.prototype.limit = 0;

        /**
         * ListBentoRequest order_by.
         * @member {bentoml.ListBentoRequest.SORTABLE_COLUMN} order_by
         * @memberof bentoml.ListBentoRequest
         * @instance
         */
        ListBentoRequest.prototype.order_by = 0;

        /**
         * ListBentoRequest ascending_order.
         * @member {boolean} ascending_order
         * @memberof bentoml.ListBentoRequest
         * @instance
         */
        ListBentoRequest.prototype.ascending_order = false;

        /**
         * Creates a new ListBentoRequest instance using the specified properties.
         * @function create
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {bentoml.IListBentoRequest=} [properties] Properties to set
         * @returns {bentoml.ListBentoRequest} ListBentoRequest instance
         */
        ListBentoRequest.create = function create(properties) {
            return new ListBentoRequest(properties);
        };

        /**
         * Encodes the specified ListBentoRequest message. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {bentoml.IListBentoRequest} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoRequest.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.bento_name != null && Object.hasOwnProperty.call(message, "bento_name"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.bento_name);
            if (message.offset != null && Object.hasOwnProperty.call(message, "offset"))
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.offset);
            if (message.limit != null && Object.hasOwnProperty.call(message, "limit"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.limit);
            if (message.order_by != null && Object.hasOwnProperty.call(message, "order_by"))
                writer.uint32(/* id 4, wireType 0 =*/32).int32(message.order_by);
            if (message.ascending_order != null && Object.hasOwnProperty.call(message, "ascending_order"))
                writer.uint32(/* id 5, wireType 0 =*/40).bool(message.ascending_order);
            return writer;
        };

        /**
         * Encodes the specified ListBentoRequest message, length delimited. Does not implicitly {@link bentoml.ListBentoRequest.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {bentoml.IListBentoRequest} message ListBentoRequest message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoRequest.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListBentoRequest message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ListBentoRequest
         * @static
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
                    message.order_by = reader.int32();
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
         * @function decodeDelimited
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoRequest} ListBentoRequest
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoRequest.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListBentoRequest message.
         * @function verify
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ListBentoRequest.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.bento_name != null && message.hasOwnProperty("bento_name"))
                if (!$util.isString(message.bento_name))
                    return "bento_name: string expected";
            if (message.offset != null && message.hasOwnProperty("offset"))
                if (!$util.isInteger(message.offset))
                    return "offset: integer expected";
            if (message.limit != null && message.hasOwnProperty("limit"))
                if (!$util.isInteger(message.limit))
                    return "limit: integer expected";
            if (message.order_by != null && message.hasOwnProperty("order_by"))
                switch (message.order_by) {
                default:
                    return "order_by: enum value expected";
                case 0:
                case 1:
                    break;
                }
            if (message.ascending_order != null && message.hasOwnProperty("ascending_order"))
                if (typeof message.ascending_order !== "boolean")
                    return "ascending_order: boolean expected";
            return null;
        };

        /**
         * Creates a ListBentoRequest message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.ListBentoRequest
         * @static
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
         * Creates a plain object from a ListBentoRequest message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ListBentoRequest
         * @static
         * @param {bentoml.ListBentoRequest} message ListBentoRequest
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ListBentoRequest to JSON.
         * @function toJSON
         * @memberof bentoml.ListBentoRequest
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        ListBentoRequest.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * SORTABLE_COLUMN enum.
         * @name bentoml.ListBentoRequest.SORTABLE_COLUMN
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
         * @memberof bentoml
         * @interface IListBentoResponse
         * @property {bentoml.IStatus|null} [status] ListBentoResponse status
         * @property {Array.<bentoml.IBento>|null} [bentos] ListBentoResponse bentos
         */

        /**
         * Constructs a new ListBentoResponse.
         * @memberof bentoml
         * @classdesc Represents a ListBentoResponse.
         * @implements IListBentoResponse
         * @constructor
         * @param {bentoml.IListBentoResponse=} [properties] Properties to set
         */
        function ListBentoResponse(properties) {
            this.bentos = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * ListBentoResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.ListBentoResponse
         * @instance
         */
        ListBentoResponse.prototype.status = null;

        /**
         * ListBentoResponse bentos.
         * @member {Array.<bentoml.IBento>} bentos
         * @memberof bentoml.ListBentoResponse
         * @instance
         */
        ListBentoResponse.prototype.bentos = $util.emptyArray;

        /**
         * Creates a new ListBentoResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {bentoml.IListBentoResponse=} [properties] Properties to set
         * @returns {bentoml.ListBentoResponse} ListBentoResponse instance
         */
        ListBentoResponse.create = function create(properties) {
            return new ListBentoResponse(properties);
        };

        /**
         * Encodes the specified ListBentoResponse message. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {bentoml.IListBentoResponse} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.bentos != null && message.bentos.length)
                for (let i = 0; i < message.bentos.length; ++i)
                    $root.bentoml.Bento.encode(message.bentos[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified ListBentoResponse message, length delimited. Does not implicitly {@link bentoml.ListBentoResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {bentoml.IListBentoResponse} message ListBentoResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        ListBentoResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a ListBentoResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.ListBentoResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.ListBentoResponse} ListBentoResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        ListBentoResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a ListBentoResponse message.
         * @function verify
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        ListBentoResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.bentos != null && message.hasOwnProperty("bentos")) {
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
         * @function fromObject
         * @memberof bentoml.ListBentoResponse
         * @static
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
         * Creates a plain object from a ListBentoResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.ListBentoResponse
         * @static
         * @param {bentoml.ListBentoResponse} message ListBentoResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this ListBentoResponse to JSON.
         * @function toJSON
         * @memberof bentoml.ListBentoResponse
         * @instance
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
         * @memberof bentoml
         * @classdesc Represents a Yatai
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
         * @function create
         * @memberof bentoml.Yatai
         * @static
         * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
         * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
         * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
         * @returns {Yatai} RPC service. Useful where requests and/or responses are streamed.
         */
        Yatai.create = function create(rpcImpl, requestDelimited, responseDelimited) {
            return new this(rpcImpl, requestDelimited, responseDelimited);
        };

        /**
         * Callback as used by {@link bentoml.Yatai#healthCheck}.
         * @memberof bentoml.Yatai
         * @typedef HealthCheckCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.HealthCheckResponse} [response] HealthCheckResponse
         */

        /**
         * Calls HealthCheck.
         * @function healthCheck
         * @memberof bentoml.Yatai
         * @instance
         * @param {google.protobuf.IEmpty} request Empty message or plain object
         * @param {bentoml.Yatai.HealthCheckCallback} callback Node-style callback called with the error, if any, and HealthCheckResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.healthCheck = function healthCheck(request, callback) {
            return this.rpcCall(healthCheck, $root.google.protobuf.Empty, $root.bentoml.HealthCheckResponse, request, callback);
        }, "name", { value: "HealthCheck" });

        /**
         * Calls HealthCheck.
         * @function healthCheck
         * @memberof bentoml.Yatai
         * @instance
         * @param {google.protobuf.IEmpty} request Empty message or plain object
         * @returns {Promise<bentoml.HealthCheckResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#getYataiServiceVersion}.
         * @memberof bentoml.Yatai
         * @typedef GetYataiServiceVersionCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.GetYataiServiceVersionResponse} [response] GetYataiServiceVersionResponse
         */

        /**
         * Calls GetYataiServiceVersion.
         * @function getYataiServiceVersion
         * @memberof bentoml.Yatai
         * @instance
         * @param {google.protobuf.IEmpty} request Empty message or plain object
         * @param {bentoml.Yatai.GetYataiServiceVersionCallback} callback Node-style callback called with the error, if any, and GetYataiServiceVersionResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.getYataiServiceVersion = function getYataiServiceVersion(request, callback) {
            return this.rpcCall(getYataiServiceVersion, $root.google.protobuf.Empty, $root.bentoml.GetYataiServiceVersionResponse, request, callback);
        }, "name", { value: "GetYataiServiceVersion" });

        /**
         * Calls GetYataiServiceVersion.
         * @function getYataiServiceVersion
         * @memberof bentoml.Yatai
         * @instance
         * @param {google.protobuf.IEmpty} request Empty message or plain object
         * @returns {Promise<bentoml.GetYataiServiceVersionResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#applyDeployment}.
         * @memberof bentoml.Yatai
         * @typedef ApplyDeploymentCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.ApplyDeploymentResponse} [response] ApplyDeploymentResponse
         */

        /**
         * Calls ApplyDeployment.
         * @function applyDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IApplyDeploymentRequest} request ApplyDeploymentRequest message or plain object
         * @param {bentoml.Yatai.ApplyDeploymentCallback} callback Node-style callback called with the error, if any, and ApplyDeploymentResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.applyDeployment = function applyDeployment(request, callback) {
            return this.rpcCall(applyDeployment, $root.bentoml.ApplyDeploymentRequest, $root.bentoml.ApplyDeploymentResponse, request, callback);
        }, "name", { value: "ApplyDeployment" });

        /**
         * Calls ApplyDeployment.
         * @function applyDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IApplyDeploymentRequest} request ApplyDeploymentRequest message or plain object
         * @returns {Promise<bentoml.ApplyDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#deleteDeployment}.
         * @memberof bentoml.Yatai
         * @typedef DeleteDeploymentCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.DeleteDeploymentResponse} [response] DeleteDeploymentResponse
         */

        /**
         * Calls DeleteDeployment.
         * @function deleteDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDeleteDeploymentRequest} request DeleteDeploymentRequest message or plain object
         * @param {bentoml.Yatai.DeleteDeploymentCallback} callback Node-style callback called with the error, if any, and DeleteDeploymentResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.deleteDeployment = function deleteDeployment(request, callback) {
            return this.rpcCall(deleteDeployment, $root.bentoml.DeleteDeploymentRequest, $root.bentoml.DeleteDeploymentResponse, request, callback);
        }, "name", { value: "DeleteDeployment" });

        /**
         * Calls DeleteDeployment.
         * @function deleteDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDeleteDeploymentRequest} request DeleteDeploymentRequest message or plain object
         * @returns {Promise<bentoml.DeleteDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#getDeployment}.
         * @memberof bentoml.Yatai
         * @typedef GetDeploymentCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.GetDeploymentResponse} [response] GetDeploymentResponse
         */

        /**
         * Calls GetDeployment.
         * @function getDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IGetDeploymentRequest} request GetDeploymentRequest message or plain object
         * @param {bentoml.Yatai.GetDeploymentCallback} callback Node-style callback called with the error, if any, and GetDeploymentResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.getDeployment = function getDeployment(request, callback) {
            return this.rpcCall(getDeployment, $root.bentoml.GetDeploymentRequest, $root.bentoml.GetDeploymentResponse, request, callback);
        }, "name", { value: "GetDeployment" });

        /**
         * Calls GetDeployment.
         * @function getDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IGetDeploymentRequest} request GetDeploymentRequest message or plain object
         * @returns {Promise<bentoml.GetDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#describeDeployment}.
         * @memberof bentoml.Yatai
         * @typedef DescribeDeploymentCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.DescribeDeploymentResponse} [response] DescribeDeploymentResponse
         */

        /**
         * Calls DescribeDeployment.
         * @function describeDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDescribeDeploymentRequest} request DescribeDeploymentRequest message or plain object
         * @param {bentoml.Yatai.DescribeDeploymentCallback} callback Node-style callback called with the error, if any, and DescribeDeploymentResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.describeDeployment = function describeDeployment(request, callback) {
            return this.rpcCall(describeDeployment, $root.bentoml.DescribeDeploymentRequest, $root.bentoml.DescribeDeploymentResponse, request, callback);
        }, "name", { value: "DescribeDeployment" });

        /**
         * Calls DescribeDeployment.
         * @function describeDeployment
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDescribeDeploymentRequest} request DescribeDeploymentRequest message or plain object
         * @returns {Promise<bentoml.DescribeDeploymentResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#listDeployments}.
         * @memberof bentoml.Yatai
         * @typedef ListDeploymentsCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.ListDeploymentsResponse} [response] ListDeploymentsResponse
         */

        /**
         * Calls ListDeployments.
         * @function listDeployments
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IListDeploymentsRequest} request ListDeploymentsRequest message or plain object
         * @param {bentoml.Yatai.ListDeploymentsCallback} callback Node-style callback called with the error, if any, and ListDeploymentsResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.listDeployments = function listDeployments(request, callback) {
            return this.rpcCall(listDeployments, $root.bentoml.ListDeploymentsRequest, $root.bentoml.ListDeploymentsResponse, request, callback);
        }, "name", { value: "ListDeployments" });

        /**
         * Calls ListDeployments.
         * @function listDeployments
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IListDeploymentsRequest} request ListDeploymentsRequest message or plain object
         * @returns {Promise<bentoml.ListDeploymentsResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#addBento}.
         * @memberof bentoml.Yatai
         * @typedef AddBentoCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.AddBentoResponse} [response] AddBentoResponse
         */

        /**
         * Calls AddBento.
         * @function addBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IAddBentoRequest} request AddBentoRequest message or plain object
         * @param {bentoml.Yatai.AddBentoCallback} callback Node-style callback called with the error, if any, and AddBentoResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.addBento = function addBento(request, callback) {
            return this.rpcCall(addBento, $root.bentoml.AddBentoRequest, $root.bentoml.AddBentoResponse, request, callback);
        }, "name", { value: "AddBento" });

        /**
         * Calls AddBento.
         * @function addBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IAddBentoRequest} request AddBentoRequest message or plain object
         * @returns {Promise<bentoml.AddBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#updateBento}.
         * @memberof bentoml.Yatai
         * @typedef UpdateBentoCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.UpdateBentoResponse} [response] UpdateBentoResponse
         */

        /**
         * Calls UpdateBento.
         * @function updateBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IUpdateBentoRequest} request UpdateBentoRequest message or plain object
         * @param {bentoml.Yatai.UpdateBentoCallback} callback Node-style callback called with the error, if any, and UpdateBentoResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.updateBento = function updateBento(request, callback) {
            return this.rpcCall(updateBento, $root.bentoml.UpdateBentoRequest, $root.bentoml.UpdateBentoResponse, request, callback);
        }, "name", { value: "UpdateBento" });

        /**
         * Calls UpdateBento.
         * @function updateBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IUpdateBentoRequest} request UpdateBentoRequest message or plain object
         * @returns {Promise<bentoml.UpdateBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#getBento}.
         * @memberof bentoml.Yatai
         * @typedef GetBentoCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.GetBentoResponse} [response] GetBentoResponse
         */

        /**
         * Calls GetBento.
         * @function getBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IGetBentoRequest} request GetBentoRequest message or plain object
         * @param {bentoml.Yatai.GetBentoCallback} callback Node-style callback called with the error, if any, and GetBentoResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.getBento = function getBento(request, callback) {
            return this.rpcCall(getBento, $root.bentoml.GetBentoRequest, $root.bentoml.GetBentoResponse, request, callback);
        }, "name", { value: "GetBento" });

        /**
         * Calls GetBento.
         * @function getBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IGetBentoRequest} request GetBentoRequest message or plain object
         * @returns {Promise<bentoml.GetBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#dangerouslyDeleteBento}.
         * @memberof bentoml.Yatai
         * @typedef DangerouslyDeleteBentoCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.DangerouslyDeleteBentoResponse} [response] DangerouslyDeleteBentoResponse
         */

        /**
         * Calls DangerouslyDeleteBento.
         * @function dangerouslyDeleteBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDangerouslyDeleteBentoRequest} request DangerouslyDeleteBentoRequest message or plain object
         * @param {bentoml.Yatai.DangerouslyDeleteBentoCallback} callback Node-style callback called with the error, if any, and DangerouslyDeleteBentoResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.dangerouslyDeleteBento = function dangerouslyDeleteBento(request, callback) {
            return this.rpcCall(dangerouslyDeleteBento, $root.bentoml.DangerouslyDeleteBentoRequest, $root.bentoml.DangerouslyDeleteBentoResponse, request, callback);
        }, "name", { value: "DangerouslyDeleteBento" });

        /**
         * Calls DangerouslyDeleteBento.
         * @function dangerouslyDeleteBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IDangerouslyDeleteBentoRequest} request DangerouslyDeleteBentoRequest message or plain object
         * @returns {Promise<bentoml.DangerouslyDeleteBentoResponse>} Promise
         * @variation 2
         */

        /**
         * Callback as used by {@link bentoml.Yatai#listBento}.
         * @memberof bentoml.Yatai
         * @typedef ListBentoCallback
         * @type {function}
         * @param {Error|null} error Error, if any
         * @param {bentoml.ListBentoResponse} [response] ListBentoResponse
         */

        /**
         * Calls ListBento.
         * @function listBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IListBentoRequest} request ListBentoRequest message or plain object
         * @param {bentoml.Yatai.ListBentoCallback} callback Node-style callback called with the error, if any, and ListBentoResponse
         * @returns {undefined}
         * @variation 1
         */
        Object.defineProperty(Yatai.prototype.listBento = function listBento(request, callback) {
            return this.rpcCall(listBento, $root.bentoml.ListBentoRequest, $root.bentoml.ListBentoResponse, request, callback);
        }, "name", { value: "ListBento" });

        /**
         * Calls ListBento.
         * @function listBento
         * @memberof bentoml.Yatai
         * @instance
         * @param {bentoml.IListBentoRequest} request ListBentoRequest message or plain object
         * @returns {Promise<bentoml.ListBentoResponse>} Promise
         * @variation 2
         */

        return Yatai;
    })();

    bentoml.HealthCheckResponse = (function() {

        /**
         * Properties of a HealthCheckResponse.
         * @memberof bentoml
         * @interface IHealthCheckResponse
         * @property {bentoml.IStatus|null} [status] HealthCheckResponse status
         */

        /**
         * Constructs a new HealthCheckResponse.
         * @memberof bentoml
         * @classdesc Represents a HealthCheckResponse.
         * @implements IHealthCheckResponse
         * @constructor
         * @param {bentoml.IHealthCheckResponse=} [properties] Properties to set
         */
        function HealthCheckResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * HealthCheckResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.HealthCheckResponse
         * @instance
         */
        HealthCheckResponse.prototype.status = null;

        /**
         * Creates a new HealthCheckResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {bentoml.IHealthCheckResponse=} [properties] Properties to set
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse instance
         */
        HealthCheckResponse.create = function create(properties) {
            return new HealthCheckResponse(properties);
        };

        /**
         * Encodes the specified HealthCheckResponse message. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {bentoml.IHealthCheckResponse} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        HealthCheckResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified HealthCheckResponse message, length delimited. Does not implicitly {@link bentoml.HealthCheckResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {bentoml.IHealthCheckResponse} message HealthCheckResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        HealthCheckResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a HealthCheckResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.HealthCheckResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.HealthCheckResponse} HealthCheckResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        HealthCheckResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a HealthCheckResponse message.
         * @function verify
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        HealthCheckResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            return null;
        };

        /**
         * Creates a HealthCheckResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.HealthCheckResponse
         * @static
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
         * Creates a plain object from a HealthCheckResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.HealthCheckResponse
         * @static
         * @param {bentoml.HealthCheckResponse} message HealthCheckResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this HealthCheckResponse to JSON.
         * @function toJSON
         * @memberof bentoml.HealthCheckResponse
         * @instance
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
         * @memberof bentoml
         * @interface IGetYataiServiceVersionResponse
         * @property {bentoml.IStatus|null} [status] GetYataiServiceVersionResponse status
         * @property {string|null} [version] GetYataiServiceVersionResponse version
         */

        /**
         * Constructs a new GetYataiServiceVersionResponse.
         * @memberof bentoml
         * @classdesc Represents a GetYataiServiceVersionResponse.
         * @implements IGetYataiServiceVersionResponse
         * @constructor
         * @param {bentoml.IGetYataiServiceVersionResponse=} [properties] Properties to set
         */
        function GetYataiServiceVersionResponse(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * GetYataiServiceVersionResponse status.
         * @member {bentoml.IStatus|null|undefined} status
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @instance
         */
        GetYataiServiceVersionResponse.prototype.status = null;

        /**
         * GetYataiServiceVersionResponse version.
         * @member {string} version
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @instance
         */
        GetYataiServiceVersionResponse.prototype.version = "";

        /**
         * Creates a new GetYataiServiceVersionResponse instance using the specified properties.
         * @function create
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {bentoml.IGetYataiServiceVersionResponse=} [properties] Properties to set
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse instance
         */
        GetYataiServiceVersionResponse.create = function create(properties) {
            return new GetYataiServiceVersionResponse(properties);
        };

        /**
         * Encodes the specified GetYataiServiceVersionResponse message. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @function encode
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {bentoml.IGetYataiServiceVersionResponse} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetYataiServiceVersionResponse.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                $root.bentoml.Status.encode(message.status, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.version != null && Object.hasOwnProperty.call(message, "version"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.version);
            return writer;
        };

        /**
         * Encodes the specified GetYataiServiceVersionResponse message, length delimited. Does not implicitly {@link bentoml.GetYataiServiceVersionResponse.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {bentoml.IGetYataiServiceVersionResponse} message GetYataiServiceVersionResponse message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        GetYataiServiceVersionResponse.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a GetYataiServiceVersionResponse message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.GetYataiServiceVersionResponse} GetYataiServiceVersionResponse
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        GetYataiServiceVersionResponse.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a GetYataiServiceVersionResponse message.
         * @function verify
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        GetYataiServiceVersionResponse.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.status != null && message.hasOwnProperty("status")) {
                let error = $root.bentoml.Status.verify(message.status);
                if (error)
                    return "status." + error;
            }
            if (message.version != null && message.hasOwnProperty("version"))
                if (!$util.isString(message.version))
                    return "version: string expected";
            return null;
        };

        /**
         * Creates a GetYataiServiceVersionResponse message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
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
         * Creates a plain object from a GetYataiServiceVersionResponse message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @static
         * @param {bentoml.GetYataiServiceVersionResponse} message GetYataiServiceVersionResponse
         * @param {$protobuf.IConversionOptions} [options] Conversion options
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
         * Converts this GetYataiServiceVersionResponse to JSON.
         * @function toJSON
         * @memberof bentoml.GetYataiServiceVersionResponse
         * @instance
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
         * @memberof bentoml
         * @interface IChunk
         * @property {Uint8Array|null} [content] Chunk content
         */

        /**
         * Constructs a new Chunk.
         * @memberof bentoml
         * @classdesc Represents a Chunk.
         * @implements IChunk
         * @constructor
         * @param {bentoml.IChunk=} [properties] Properties to set
         */
        function Chunk(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * Chunk content.
         * @member {Uint8Array} content
         * @memberof bentoml.Chunk
         * @instance
         */
        Chunk.prototype.content = $util.newBuffer([]);

        /**
         * Creates a new Chunk instance using the specified properties.
         * @function create
         * @memberof bentoml.Chunk
         * @static
         * @param {bentoml.IChunk=} [properties] Properties to set
         * @returns {bentoml.Chunk} Chunk instance
         */
        Chunk.create = function create(properties) {
            return new Chunk(properties);
        };

        /**
         * Encodes the specified Chunk message. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @function encode
         * @memberof bentoml.Chunk
         * @static
         * @param {bentoml.IChunk} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Chunk.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.content != null && Object.hasOwnProperty.call(message, "content"))
                writer.uint32(/* id 1, wireType 2 =*/10).bytes(message.content);
            return writer;
        };

        /**
         * Encodes the specified Chunk message, length delimited. Does not implicitly {@link bentoml.Chunk.verify|verify} messages.
         * @function encodeDelimited
         * @memberof bentoml.Chunk
         * @static
         * @param {bentoml.IChunk} message Chunk message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        Chunk.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a Chunk message from the specified reader or buffer.
         * @function decode
         * @memberof bentoml.Chunk
         * @static
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
         * @function decodeDelimited
         * @memberof bentoml.Chunk
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {bentoml.Chunk} Chunk
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        Chunk.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a Chunk message.
         * @function verify
         * @memberof bentoml.Chunk
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        Chunk.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.content != null && message.hasOwnProperty("content"))
                if (!(message.content && typeof message.content.length === "number" || $util.isString(message.content)))
                    return "content: buffer expected";
            return null;
        };

        /**
         * Creates a Chunk message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof bentoml.Chunk
         * @static
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
         * Creates a plain object from a Chunk message. Also converts values to other types if specified.
         * @function toObject
         * @memberof bentoml.Chunk
         * @static
         * @param {bentoml.Chunk} message Chunk
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        Chunk.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults)
                if (options.bytes === String)
                    object.content = "";
                else {
                    object.content = [];
                    if (options.bytes !== Array)
                        object.content = $util.newBuffer(object.content);
                }
            if (message.content != null && message.hasOwnProperty("content"))
                object.content = options.bytes === String ? $util.base64.encode(message.content, 0, message.content.length) : options.bytes === Array ? Array.prototype.slice.call(message.content) : message.content;
            return object;
        };

        /**
         * Converts this Chunk to JSON.
         * @function toJSON
         * @memberof bentoml.Chunk
         * @instance
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
         * @memberof google
         * @namespace
         */
        const protobuf = {};

        protobuf.Struct = (function() {

            /**
             * Properties of a Struct.
             * @memberof google.protobuf
             * @interface IStruct
             * @property {Object.<string,google.protobuf.IValue>|null} [fields] Struct fields
             */

            /**
             * Constructs a new Struct.
             * @memberof google.protobuf
             * @classdesc Represents a Struct.
             * @implements IStruct
             * @constructor
             * @param {google.protobuf.IStruct=} [properties] Properties to set
             */
            function Struct(properties) {
                this.fields = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Struct fields.
             * @member {Object.<string,google.protobuf.IValue>} fields
             * @memberof google.protobuf.Struct
             * @instance
             */
            Struct.prototype.fields = $util.emptyObject;

            /**
             * Creates a new Struct instance using the specified properties.
             * @function create
             * @memberof google.protobuf.Struct
             * @static
             * @param {google.protobuf.IStruct=} [properties] Properties to set
             * @returns {google.protobuf.Struct} Struct instance
             */
            Struct.create = function create(properties) {
                return new Struct(properties);
            };

            /**
             * Encodes the specified Struct message. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @function encode
             * @memberof google.protobuf.Struct
             * @static
             * @param {google.protobuf.IStruct} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Struct.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.fields != null && Object.hasOwnProperty.call(message, "fields"))
                    for (let keys = Object.keys(message.fields), i = 0; i < keys.length; ++i) {
                        writer.uint32(/* id 1, wireType 2 =*/10).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]);
                        $root.google.protobuf.Value.encode(message.fields[keys[i]], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim().ldelim();
                    }
                return writer;
            };

            /**
             * Encodes the specified Struct message, length delimited. Does not implicitly {@link google.protobuf.Struct.verify|verify} messages.
             * @function encodeDelimited
             * @memberof google.protobuf.Struct
             * @static
             * @param {google.protobuf.IStruct} message Struct message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Struct.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Struct message from the specified reader or buffer.
             * @function decode
             * @memberof google.protobuf.Struct
             * @static
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
             * @function decodeDelimited
             * @memberof google.protobuf.Struct
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Struct} Struct
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Struct.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Struct message.
             * @function verify
             * @memberof google.protobuf.Struct
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Struct.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.fields != null && message.hasOwnProperty("fields")) {
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
             * @function fromObject
             * @memberof google.protobuf.Struct
             * @static
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
             * Creates a plain object from a Struct message. Also converts values to other types if specified.
             * @function toObject
             * @memberof google.protobuf.Struct
             * @static
             * @param {google.protobuf.Struct} message Struct
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this Struct to JSON.
             * @function toJSON
             * @memberof google.protobuf.Struct
             * @instance
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
             * @memberof google.protobuf
             * @interface IValue
             * @property {google.protobuf.NullValue|null} [nullValue] Value nullValue
             * @property {number|null} [numberValue] Value numberValue
             * @property {string|null} [stringValue] Value stringValue
             * @property {boolean|null} [boolValue] Value boolValue
             * @property {google.protobuf.IStruct|null} [structValue] Value structValue
             * @property {google.protobuf.IListValue|null} [listValue] Value listValue
             */

            /**
             * Constructs a new Value.
             * @memberof google.protobuf
             * @classdesc Represents a Value.
             * @implements IValue
             * @constructor
             * @param {google.protobuf.IValue=} [properties] Properties to set
             */
            function Value(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Value nullValue.
             * @member {google.protobuf.NullValue} nullValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.nullValue = 0;

            /**
             * Value numberValue.
             * @member {number} numberValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.numberValue = 0;

            /**
             * Value stringValue.
             * @member {string} stringValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.stringValue = "";

            /**
             * Value boolValue.
             * @member {boolean} boolValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.boolValue = false;

            /**
             * Value structValue.
             * @member {google.protobuf.IStruct|null|undefined} structValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.structValue = null;

            /**
             * Value listValue.
             * @member {google.protobuf.IListValue|null|undefined} listValue
             * @memberof google.protobuf.Value
             * @instance
             */
            Value.prototype.listValue = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Value kind.
             * @member {"nullValue"|"numberValue"|"stringValue"|"boolValue"|"structValue"|"listValue"|undefined} kind
             * @memberof google.protobuf.Value
             * @instance
             */
            Object.defineProperty(Value.prototype, "kind", {
                get: $util.oneOfGetter($oneOfFields = ["nullValue", "numberValue", "stringValue", "boolValue", "structValue", "listValue"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Value instance using the specified properties.
             * @function create
             * @memberof google.protobuf.Value
             * @static
             * @param {google.protobuf.IValue=} [properties] Properties to set
             * @returns {google.protobuf.Value} Value instance
             */
            Value.create = function create(properties) {
                return new Value(properties);
            };

            /**
             * Encodes the specified Value message. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @function encode
             * @memberof google.protobuf.Value
             * @static
             * @param {google.protobuf.IValue} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Value.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nullValue != null && Object.hasOwnProperty.call(message, "nullValue"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int32(message.nullValue);
                if (message.numberValue != null && Object.hasOwnProperty.call(message, "numberValue"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.numberValue);
                if (message.stringValue != null && Object.hasOwnProperty.call(message, "stringValue"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.stringValue);
                if (message.boolValue != null && Object.hasOwnProperty.call(message, "boolValue"))
                    writer.uint32(/* id 4, wireType 0 =*/32).bool(message.boolValue);
                if (message.structValue != null && Object.hasOwnProperty.call(message, "structValue"))
                    $root.google.protobuf.Struct.encode(message.structValue, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.listValue != null && Object.hasOwnProperty.call(message, "listValue"))
                    $root.google.protobuf.ListValue.encode(message.listValue, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified Value message, length delimited. Does not implicitly {@link google.protobuf.Value.verify|verify} messages.
             * @function encodeDelimited
             * @memberof google.protobuf.Value
             * @static
             * @param {google.protobuf.IValue} message Value message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Value.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Value message from the specified reader or buffer.
             * @function decode
             * @memberof google.protobuf.Value
             * @static
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
                        message.nullValue = reader.int32();
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
             * @function decodeDelimited
             * @memberof google.protobuf.Value
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Value} Value
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Value.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Value message.
             * @function verify
             * @memberof google.protobuf.Value
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Value.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let properties = {};
                if (message.nullValue != null && message.hasOwnProperty("nullValue")) {
                    properties.kind = 1;
                    switch (message.nullValue) {
                    default:
                        return "nullValue: enum value expected";
                    case 0:
                        break;
                    }
                }
                if (message.numberValue != null && message.hasOwnProperty("numberValue")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (typeof message.numberValue !== "number")
                        return "numberValue: number expected";
                }
                if (message.stringValue != null && message.hasOwnProperty("stringValue")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (!$util.isString(message.stringValue))
                        return "stringValue: string expected";
                }
                if (message.boolValue != null && message.hasOwnProperty("boolValue")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    if (typeof message.boolValue !== "boolean")
                        return "boolValue: boolean expected";
                }
                if (message.structValue != null && message.hasOwnProperty("structValue")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        let error = $root.google.protobuf.Struct.verify(message.structValue);
                        if (error)
                            return "structValue." + error;
                    }
                }
                if (message.listValue != null && message.hasOwnProperty("listValue")) {
                    if (properties.kind === 1)
                        return "kind: multiple values";
                    properties.kind = 1;
                    {
                        let error = $root.google.protobuf.ListValue.verify(message.listValue);
                        if (error)
                            return "listValue." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a Value message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof google.protobuf.Value
             * @static
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
             * Creates a plain object from a Value message. Also converts values to other types if specified.
             * @function toObject
             * @memberof google.protobuf.Value
             * @static
             * @param {google.protobuf.Value} message Value
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
                    object.numberValue = options.json && !isFinite(message.numberValue) ? String(message.numberValue) : message.numberValue;
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
             * Converts this Value to JSON.
             * @function toJSON
             * @memberof google.protobuf.Value
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Value.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            return Value;
        })();

        /**
         * NullValue enum.
         * @name google.protobuf.NullValue
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
             * @memberof google.protobuf
             * @interface IListValue
             * @property {Array.<google.protobuf.IValue>|null} [values] ListValue values
             */

            /**
             * Constructs a new ListValue.
             * @memberof google.protobuf
             * @classdesc Represents a ListValue.
             * @implements IListValue
             * @constructor
             * @param {google.protobuf.IListValue=} [properties] Properties to set
             */
            function ListValue(properties) {
                this.values = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ListValue values.
             * @member {Array.<google.protobuf.IValue>} values
             * @memberof google.protobuf.ListValue
             * @instance
             */
            ListValue.prototype.values = $util.emptyArray;

            /**
             * Creates a new ListValue instance using the specified properties.
             * @function create
             * @memberof google.protobuf.ListValue
             * @static
             * @param {google.protobuf.IListValue=} [properties] Properties to set
             * @returns {google.protobuf.ListValue} ListValue instance
             */
            ListValue.create = function create(properties) {
                return new ListValue(properties);
            };

            /**
             * Encodes the specified ListValue message. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @function encode
             * @memberof google.protobuf.ListValue
             * @static
             * @param {google.protobuf.IListValue} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ListValue.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.values != null && message.values.length)
                    for (let i = 0; i < message.values.length; ++i)
                        $root.google.protobuf.Value.encode(message.values[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified ListValue message, length delimited. Does not implicitly {@link google.protobuf.ListValue.verify|verify} messages.
             * @function encodeDelimited
             * @memberof google.protobuf.ListValue
             * @static
             * @param {google.protobuf.IListValue} message ListValue message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ListValue.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a ListValue message from the specified reader or buffer.
             * @function decode
             * @memberof google.protobuf.ListValue
             * @static
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
             * @function decodeDelimited
             * @memberof google.protobuf.ListValue
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.ListValue} ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ListValue.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a ListValue message.
             * @function verify
             * @memberof google.protobuf.ListValue
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ListValue.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.values != null && message.hasOwnProperty("values")) {
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
             * @function fromObject
             * @memberof google.protobuf.ListValue
             * @static
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
             * Creates a plain object from a ListValue message. Also converts values to other types if specified.
             * @function toObject
             * @memberof google.protobuf.ListValue
             * @static
             * @param {google.protobuf.ListValue} message ListValue
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this ListValue to JSON.
             * @function toJSON
             * @memberof google.protobuf.ListValue
             * @instance
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
             * @memberof google.protobuf
             * @interface ITimestamp
             * @property {number|null} [seconds] Timestamp seconds
             * @property {number|null} [nanos] Timestamp nanos
             */

            /**
             * Constructs a new Timestamp.
             * @memberof google.protobuf
             * @classdesc Represents a Timestamp.
             * @implements ITimestamp
             * @constructor
             * @param {google.protobuf.ITimestamp=} [properties] Properties to set
             */
            function Timestamp(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Timestamp seconds.
             * @member {number} seconds
             * @memberof google.protobuf.Timestamp
             * @instance
             */
            Timestamp.prototype.seconds = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * Timestamp nanos.
             * @member {number} nanos
             * @memberof google.protobuf.Timestamp
             * @instance
             */
            Timestamp.prototype.nanos = 0;

            /**
             * Creates a new Timestamp instance using the specified properties.
             * @function create
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {google.protobuf.ITimestamp=} [properties] Properties to set
             * @returns {google.protobuf.Timestamp} Timestamp instance
             */
            Timestamp.create = function create(properties) {
                return new Timestamp(properties);
            };

            /**
             * Encodes the specified Timestamp message. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @function encode
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {google.protobuf.ITimestamp} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Timestamp.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.seconds != null && Object.hasOwnProperty.call(message, "seconds"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.seconds);
                if (message.nanos != null && Object.hasOwnProperty.call(message, "nanos"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.nanos);
                return writer;
            };

            /**
             * Encodes the specified Timestamp message, length delimited. Does not implicitly {@link google.protobuf.Timestamp.verify|verify} messages.
             * @function encodeDelimited
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {google.protobuf.ITimestamp} message Timestamp message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Timestamp.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Timestamp message from the specified reader or buffer.
             * @function decode
             * @memberof google.protobuf.Timestamp
             * @static
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
             * @function decodeDelimited
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Timestamp} Timestamp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Timestamp.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Timestamp message.
             * @function verify
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Timestamp.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.seconds != null && message.hasOwnProperty("seconds"))
                    if (!$util.isInteger(message.seconds) && !(message.seconds && $util.isInteger(message.seconds.low) && $util.isInteger(message.seconds.high)))
                        return "seconds: integer|Long expected";
                if (message.nanos != null && message.hasOwnProperty("nanos"))
                    if (!$util.isInteger(message.nanos))
                        return "nanos: integer expected";
                return null;
            };

            /**
             * Creates a Timestamp message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof google.protobuf.Timestamp
             * @static
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
             * Creates a plain object from a Timestamp message. Also converts values to other types if specified.
             * @function toObject
             * @memberof google.protobuf.Timestamp
             * @static
             * @param {google.protobuf.Timestamp} message Timestamp
             * @param {$protobuf.IConversionOptions} [options] Conversion options
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
             * Converts this Timestamp to JSON.
             * @function toJSON
             * @memberof google.protobuf.Timestamp
             * @instance
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
             * @memberof google.protobuf
             * @interface IEmpty
             */

            /**
             * Constructs a new Empty.
             * @memberof google.protobuf
             * @classdesc Represents an Empty.
             * @implements IEmpty
             * @constructor
             * @param {google.protobuf.IEmpty=} [properties] Properties to set
             */
            function Empty(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new Empty instance using the specified properties.
             * @function create
             * @memberof google.protobuf.Empty
             * @static
             * @param {google.protobuf.IEmpty=} [properties] Properties to set
             * @returns {google.protobuf.Empty} Empty instance
             */
            Empty.create = function create(properties) {
                return new Empty(properties);
            };

            /**
             * Encodes the specified Empty message. Does not implicitly {@link google.protobuf.Empty.verify|verify} messages.
             * @function encode
             * @memberof google.protobuf.Empty
             * @static
             * @param {google.protobuf.IEmpty} message Empty message or plain object to encode
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
             * @function encodeDelimited
             * @memberof google.protobuf.Empty
             * @static
             * @param {google.protobuf.IEmpty} message Empty message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Empty.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an Empty message from the specified reader or buffer.
             * @function decode
             * @memberof google.protobuf.Empty
             * @static
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
             * @function decodeDelimited
             * @memberof google.protobuf.Empty
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {google.protobuf.Empty} Empty
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Empty.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an Empty message.
             * @function verify
             * @memberof google.protobuf.Empty
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Empty.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                return null;
            };

            /**
             * Creates an Empty message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof google.protobuf.Empty
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {google.protobuf.Empty} Empty
             */
            Empty.fromObject = function fromObject(object) {
                if (object instanceof $root.google.protobuf.Empty)
                    return object;
                return new $root.google.protobuf.Empty();
            };

            /**
             * Creates a plain object from an Empty message. Also converts values to other types if specified.
             * @function toObject
             * @memberof google.protobuf.Empty
             * @static
             * @param {google.protobuf.Empty} message Empty
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Empty.toObject = function toObject() {
                return {};
            };

            /**
             * Converts this Empty to JSON.
             * @function toJSON
             * @memberof google.protobuf.Empty
             * @instance
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
