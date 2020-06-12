import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const handlerConfigToTableContent = (
  handler_config
): Array<string> | null | undefined => {
  if (!handler_config) {
    return "None";
  }
  const config = handler_config.fields;
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
    const key = configureKeys[index];
    const valueObject = config[key];
    const valueType = Object.keys(valueObject)[0];
    let value = Object.values(valueObject)[0];
    if (valueType == "nullValue") {
      value = "null";
    }

    displayHandlerList.push(<p key={index}>{`${key}: ${value}`}</p>);
  }
  return displayHandlerList;
};

interface IApiProps {
  name: string;
  input_type: string;
  docs: string;
  handler_config: { [key: string]: string };
}

const APIS_TABLE_HEADER = [
  "API name",
  "Handler type",
  "Handler Config",
  "Documentation",
];

const APIS_TABLE_RATIO = [1, 1, 1, 4];

const ApisTable: React.FC<{ apis: Array<IApiProps> }> = ({ apis }) => {
  const apisTableContent = apis.map((api) => ({
    content: [
      api.name,
      api.input_type,
      handlerConfigToTableContent(api.handler_config),
      api.docs,
    ],
  }));
  return (
    <Section>
      <h2>APIs</h2>
      <Table
        content={apisTableContent}
        header={APIS_TABLE_HEADER}
        ratio={APIS_TABLE_RATIO}
      />
    </Section>
  );
};
export default ApisTable;
