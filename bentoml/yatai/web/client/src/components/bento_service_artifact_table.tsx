import * as React from 'react';
import { Table, Column, Cell } from '@blueprintjs/table';

export const BentoServiceArtifacts = ({artifacts}) => {
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
