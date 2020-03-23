import * as React from "react";
import { TableContainer, Row, Cell, TableHeader } from "../../ui/Table";

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

const ApisTable: React.FC<{ apis: Array<IApiProps> }> = ({ apis }) => (
  <div>
    <h2>APIs</h2>
    <TableContainer>
      <TableHeader>
        <Cell maxWidth={150}>API name</Cell>
        <Cell maxWidth={250}>Handler type</Cell>
        <Cell>Handler Config</Cell>
        <Cell maxWidth={300}>Documentation</Cell>
      </TableHeader>
      {apis.map((api, i) => (
        <Row key={i}>
          <Cell maxWidth={150}>{api.name}</Cell>
          <Cell maxWidth={250}>{api.handler_type}</Cell>
          <Cell>
            {parseHandlerConfigAsKeyValueArray(api.handler_config.fields).map(
              (field, i) => (
                <p key={i}>{field}</p>
              )
            )}
          </Cell>
          <Cell maxWidth={300}>{api.docs}</Cell>
        </Row>
      ))}
    </TableContainer>
  </div>
);
export default ApisTable;
