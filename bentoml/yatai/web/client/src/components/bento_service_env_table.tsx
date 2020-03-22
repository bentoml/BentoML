import * as React from 'react';
import { Table, Column, Cell } from '@blueprintjs/table';

export const BentoServiceEnvironments = ({env}) => {
  const envKeys = Object.keys(env);
  const renderEnvKey = (rowIndex) => (
    <Cell>{envKeys[rowIndex]}</Cell>
  );
  const renderEnvValue = (rowIndex) => (
    <Cell><pre>{env[envKeys[rowIndex]]}</pre></Cell>
  )
  return (
    <div>
      <h3>Environments</h3>
      <Table numRows={envKeys.length}>
        <Column cellRenderer={renderEnvKey} />
        <Column cellRenderer={renderEnvValue} />
      </Table>
    </div>
  )
};
