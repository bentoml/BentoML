import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const parseHandlerConfigAsKeyValueArray = (config): Array<string> => {
  /*
    grpc format for JSON:
    example
    fields {
      input_type: {
        nullValue: 'NULL_VALUE',
      },
      orient: {
        stringValue: 'frame',
      }
    }
    */

  const displayHandlerList = [];
  const configureKeys = Object.keys(config);
  for (let index = 0; index < configureKeys.length; index++) {
    let key = configureKeys[index];
    let valueObject = config[key];
    let value = Object.values(valueObject)[0];
    let valueType = Object.keys(valueObject)[0];
    if (valueType == "nullValue") {
      value = "null";
    }

    displayHandlerList.push(`${key}: ${value}`);
  }
  return displayHandlerList;
};

interface IApiProps {
  name: string;
  handler_type: string;
  docs: string;
  handler_config: { [key: string]: string };
}

const APIS_TABLE_HEADER = [
  "API name",
  "Handler type",
  "Handler Config",
  "Documentation"
];

const APIS_TABLE_RATIO = [1, 1, 3, 1];

const ApisTable: React.FC<{ apis: Array<IApiProps> }> = ({ apis }) => {
  const parsedApis = apis.map(api => [
    api.name,
    api.handler_type,
    parseHandlerConfigAsKeyValueArray(
      api.handler_config.fields
    ).map((field, i) => <p key={i}>{field}</p>),
    api.docs
  ]);
  return (
    <Section>
      <h2>APIs</h2>
      <Table
        content={parsedApis}
        header={APIS_TABLE_HEADER}
        ratio={APIS_TABLE_RATIO}
      />
    </Section>
  );
};
export default ApisTable;
