import * as React from 'react';
import { Table, Column, Cell } from '@blueprintjs/table';

import { FetchContainer } from '../utils/index';

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

const BentoServiceAPIs = ({apis}) => {
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

const BentoServiceArtifacts = ({artifacts}) => {
  const renderArtifactName = (rowIndex) => (
    <Cell>{artifacts[rowIndex].name}</Cell>
  );
  const renderArtifactType = (rowIndex) => (
    <Cell>{artifacts[rowIndex].artifact_type}</Cell>
  );

  return (
    <div>
      <h3>Artifacts</h3>
      <Table numRows={artifacts.length}>
        <Column name='Artifact name' cellRenderer={renderArtifactName} />
        <Column name='Artifact type' cellRenderer={renderArtifactType} />
      </Table>
    </div>
  )
};

const BentoServiceEnvironments = ({environments}) => {
  return (
    <div>
      environments
    </div>
  )
};

export const BentoServiceDetail = (props) => {
  const params = props.match.params;

  return (
    <FetchContainer
      url='/api/GetBento'
      params={{bento_name: params.name, bento_version: params.version}}
    >
      {
        (data, error) => {
          let displayBentoServiceDetail;
          if (error) {
            return (<div>error</div>);
          }

          if (data && data.data && data.data.bento) {
            console.log(data.data.bento);
            const bento = data.data.bento;

            displayBentoServiceDetail = (
              <div>
                <h2>{params.name}:{params.version}</h2>
                <h4></h4>
                <h4></h4>
                <BentoServiceAPIs apis={bento.bento_service_metadata.apis}/>
                <BentoServiceArtifacts
                  artifacts={bento.bento_service_metadata.artifacts}
                />
                <h3>Environments</h3>
                <BentoServiceEnvironments env={bento.bento_service_metadata.env}/>
              </div>
            )
          } else {
            displayBentoServiceDetail = (
              <div>grpc error</div>
            )
          }

          return (
            <div>
              {displayBentoServiceDetail}
            </div>
          )
        }
      }
    </FetchContainer>
  )
};