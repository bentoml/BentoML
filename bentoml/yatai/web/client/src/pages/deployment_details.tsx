import * as React from 'react';
import { Link } from 'react-router-dom';
import { Table, Column, Cell } from '@blueprintjs/table';

import { FetchContainer, displayTimeInFromNowFormat } from '../utils/index';
import { BentoServiceAPIs } from '../components/bento_service_api_table';



const DeploymentInfo = ({deployment}) => {
  const infoList = ['Created at', 'Updated at', 'BentoService', 'Endpoint'];

  const renderName = (rowIndex) => (<Cell>{infoList[rowIndex]}</Cell>);
  const renderValue = (rowIndex) => {
    const fieldName = infoList[rowIndex];
    let value;
    switch (fieldName) {
      case 'Created at':
        value = displayTimeInFromNowFormat(Number(deployment.created_at.seconds));
        break;
      case 'Updated at':
        value = displayTimeInFromNowFormat(Number(deployment.last_updated_at.seconds));
        break;
      case 'BentoService':
        const spec = deployment.spec
        const bentoTag = `${spec.bento_name}:${spec.bento_version}`;
        value = (
          <Link to={`/repository/${spec.bento_name}/${spec.bento_version}`}>
            {bentoTag}
          </Link>
        );
        break;
      case 'Endpoint':
        value = 'Not Available';
        const {state} = deployment;
        if (state.state == 'RUNNING' && state.info_json) {
          const infoJson = JSON.parse(state.info_json)
          value = (
            <pre>{infoJson.endpoints.join('\n')}</pre>
          )
        }
        break;
    }

    return (
      <Cell>{value}</Cell>
    )
  };
  return (
    <div>
      <h3>Info</h3>
      <Table numRows={infoList.length}>
        <Column cellRenderer={renderName} />
        <Column cellRenderer={renderValue} />
      </Table>
    </div>
  )
};

const DisplayDeploymentConfig = ({config}) => {
  console.log('display', config);
  const configKeys = Object.keys(config);

  const renderConfigName = (rowIndex) => {
    return (
      <Cell>{configKeys[rowIndex]}</Cell>
    )
  }

  const renderConfigValue = (rowIndex) => {
    const configValue = config[configKeys[rowIndex]];
    return (
      <Cell>{configValue}</Cell>
    )
  }

  return (
    <Table numKeys={configKeys.length}>
      <Column cellRenderer={renderConfigName} />
      <Column cellRenderer={renderConfigValue} />
    </Table>
  );
}

const DeploymentSpec = ({spec}) => {
  let config;
  switch (spec.operator) {
    case 'AWS_LAMBDA':
      config = spec.aws_lambda_operator_config;
      break;
    case 'AWS_SAGEMAKER':
      config = spec.sagemaker_operator_config;
      break;
    default:
      config = {}
  }

  return (
    <div>
      <h3>Configuration</h3>
      <DisplayDeploymentConfig config={config} />
    </div>
  )
};

const DeploymentError = () => {
  return (
    <div>
      if error display it
    </div>
  )
}


const DeploymentAPIs = ({deployment}) => {
  const {spec} = deployment;
  let config;
  switch (spec.operator) {
    case 'AWS_LAMBDA':
      config = spec.aws_lambda_operator_config;
      break;
    case 'AWS_SAGEMAKER':
      config = spec.sagemaker_operator_config;
      break;
  }
  const apiName = config.api_name;


  return (
    <FetchContainer
      url='/api/GetBento'
      params={{
        bento_name: deployment.spec.bento_name,
        bento_version: deployment.spec.bento_version,
      }}
    >
      {
        (data, error) => {
          if (error) {
            return (
              <div>
                <h3>APIs</h3>
                error
              </div>);
          }

          if (data && data.data && data.data.bento) {
            const bento = data.data.bento;
            let { apis } = bento.bento_service_metadata;
            if (apiName) {
              const deployedApi = apis.find((api) => {
                return api.name == apiName;
              });
              apis = [deployedApi];
            }
            return (<BentoServiceAPIs apis={apis}/>);
          } else {
            return (
              <div>
                <h3>APIs</h3>
                grpc error
              </div>
            )
          }
        }
      }
    </FetchContainer>
  );
}


export const DeploymentDetails = (props) => {
  const params = props.match.params;
  return (
    <FetchContainer
      url='/api/GetDeployment'
      params={{deployment_name: params.name, namespace: params.namespace}}
    >
      {
        (data, error) => {
          console.log(data, error)
          let detailDisplay;
          if (error) {
            return (<div>error</div>)
          }

          if (data.data && data.data.deployment) {
            const deployment = data.data.deployment;
            detailDisplay = (
              <div>
                <DeploymentError />
                <DeploymentInfo deployment={deployment} />
                <DeploymentSpec spec={deployment.spec} />
                <DeploymentAPIs deployment={deployment} />
              </div>
            );
          } else {
            detailDisplay = (
              <div>grpc error</div>
            )
          }
          return (
            <div>
              <div>breadcrumb</div>
              {detailDisplay}
            </div>
          );
        }
      }
    </FetchContainer>
  )
};