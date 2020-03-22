import * as React from 'react';
import { Table, Column, Cell } from '@blueprintjs/table';

const parseHandlerConfigAsKeyValueArray = (config) => {
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
  const configureKeys = Object.keys(config)
  for (let index = 0; index < configureKeys.length; index++) {
    let key = configureKeys[index];
    let valueObject = config[key];
    let value = Object.values(valueObject)[0];
    let valueType = Object.keys(valueObject)[0];
    if (valueType == 'nullValue') {
      value = 'null';
    }

    displayHandlerList.push(`${key}: ${value}`);
  }
  return displayHandlerList;
}

export const BentoServiceAPIs = ({apis}) => {
  const renderName=(rowIndex) => (
    <Cell>{apis[rowIndex].name}</Cell>
  );

  const renderHandlerType = (rowIndex) => (
    <Cell>{apis[rowIndex].handler_type}</Cell>
  );

  const renderHandlerConfig = (rowIndex) => {
    const handlerConfig = apis[rowIndex].handler_config;
    const displayHandlerList = parseHandlerConfigAsKeyValueArray(handlerConfig.fields);
    return (
      <Cell truncated={false}>
        {
          displayHandlerList.map((config) => <div>{config}</div>)
        }
      </Cell>
    )
  };

  const renderDocumentation = (rowIndex) => (
    <Cell>{apis[rowIndex].docs}</Cell>
  );

  return (
    <div>
      <h3>APIs</h3>
      <Table numRows={apis.length}>
        <Column name='API name' cellRenderer={renderName} />
        <Column name='Handler type' cellRenderer={renderHandlerType} />
        <Column name='Handler Config' cellRenderer={renderHandlerConfig} />
        <Column name='Documentation' cellRenderer={renderDocumentation} />
      </Table>
    </div>
  )
};
