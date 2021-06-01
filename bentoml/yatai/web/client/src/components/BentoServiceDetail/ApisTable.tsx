import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const inputConfigToTableContent = (
  input_config: any
): string | null | undefined | Array<any> => {
  if (!input_config) {
    return ["None"];
  }
  const config = input_config.fields;
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

  const displayInputList: React.ReactNode[] = [];
  const configureKeys = Object.keys(config);
  for (let index = 0; index < configureKeys.length; index++) {
    const key = configureKeys[index];
    const valueObject = config[key];
    const valueType = Object.keys(valueObject)[0];
    let value = Object.values(valueObject)[0];
    if (valueType === "nullValue") {
      value = "null";
    }

    displayInputList.push(<p key={index}>{`${key}: ${value}`}</p>);
  }
  return displayInputList;
};

interface IApiProps {
  name: string;
  input_type: string;
  docs: string;
  input_config: { [key: string]: string };
  output_type: string;
  batch: boolean;
  route: string;
}

const APIS_TABLE_HEADER = [
  "API name",
  "route",
  "is batch enabled",
  "Input type",
  "Input Config",
  "Output type",
  "Documentation",
];

const APIS_TABLE_RATIO = [1, 1, 1, 1, 1, 1, 4];

const ApisTable: React.FC<{ apis: Array<IApiProps> }> = ({ apis }) => {
  const apisTableContent = apis.map((api) => ({
    content: [
      api.name,
      api.route ? api.route : `/${api.name}`,
      api.batch ? "Yes" : "No",
      api.input_type,
      inputConfigToTableContent(api.input_config),
      api.output_type,
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
