import * as React from 'react';
import { TableContainer, TableHeader, Row, Cell } from '../../ui/Table';


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
  return (
    <div>
      <h2>Configuration</h2>
      <TableContainer>
        <Row>
          <Cell maxWidth={150}>Platform</Cell>
          <Cell>{spec.operator}</Cell>
        </Row>
        {
          configKeys.map((configKey, i) => {
            return (
              <Row key={i}>
                <Cell maxWidth={150}>{configKey}:</Cell>
                <Cell>{config[configKey]}</Cell>
              </Row>
            )
          })
        }
      </TableContainer>
    </div>
  );
};

export default ConfigurationTable;