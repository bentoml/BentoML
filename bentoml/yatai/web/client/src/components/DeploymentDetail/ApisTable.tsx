import * as React from 'react';
import Table from '../../ui/Table';
import { HttpRequestContainer, DisplayHttpError } from '../../utils/http_container';
import ApisTable from '../BentoServiceDetail/ApisTable';

const DeploymentApisTable = ({deployment}) => {
  let apiName;
  switch (deployment.spec.operator) {
    case 'AWS_LAMBDA':
      apiName = deployment.spec.aws_lambda_operator_config.api_name;
      break;
    case 'AWS_SAGEMAKER':
      apiName = deployment.spec.sagemaker_opeartor_config.api_name;
      break;
  }
  return (
    <HttpRequestContainer
      url="/api/GetBento"
      params={{
        bento_name: deployment.spec.bento_name,
        bento_version: deployment.spec.bento_version
      }}
    >
      {(data, isLoading, error) => {
        if (isLoading) {
          return <div>Loading...</div>
        }
        if (error) {
          return (
            <div>
              <h3>APIs</h3>
              <DisplayHttpError error={error} />
            </div>
          );
        }

        if (data && data.data && data.data.bento) {
          const bento = data.data.bento;
          let { apis } = bento.bento_service_metadata;
          if (apiName) {
            const deployedApi = apis.find(api => {
              return api.name == apiName;
            });
            apis = [deployedApi];
          }
          return <ApisTable apis={apis} />;
        } else {
          return (
            <div>
              <h3>APIs</h3>
              No APIs
            </div>
          );
        }
      }}
    </HttpRequestContainer>
  )
};

export default DeploymentApisTable;
