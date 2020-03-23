import * as React from "react";
import { TableContainer, Row, Cell, TableHeader } from "../../ui/Table";

interface IArtifactProps {
  name: string;
  artifact_type: string;
}

const ArtifactsTable: React.FC<{ artifacts: Array<IArtifactProps> }> = ({
  artifacts
}) => (
  <div>
    <h2>Artifacts</h2>
    <TableContainer>
      <TableHeader>
        <Cell maxWidth={150} color="#137CBD">
          Artifact name
        </Cell>
        <Cell>Artifact type</Cell>
      </TableHeader>
      {artifacts.map((artifact, i) => (
        <Row key={i}>
          <Cell maxWidth={150}>{artifact.name}</Cell>
          <Cell>{artifact.artifact_type}</Cell>
        </Row>
      ))}
    </TableContainer>
  </div>
);
export default ArtifactsTable;
