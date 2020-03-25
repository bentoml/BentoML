import * as React from 'react';
import Table from '../../ui/Table';


const ConfigurationTable = ({spec}) => {
  let  config;
  switch (spec.operator) {
    case "AWS_LAMBDA":
      config = spec.aws_lambda_operator_config;
      break;
    case "AWS_SAGEMAKER":
      config = spec.sagemaker_operator_config;
      break;
    default:
      config = {};
  }
  const configKeys = Object.keys(config);
  const parsedConfiguration = [
    [
      'Platform',
      spec.operator,
    ]
  ]
  configKeys.forEach(key => {
    parsedConfiguration.push([
      key,
      config[key]
    ]);
  });
  return (
    <div>
      <h2>Configuration</h2>
      <Table
         content={parsedConfiguration}
         ratio={[1, 4]}
      />
    </div>
  );
};

export default ConfigurationTable;