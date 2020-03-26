import * as React from "react";
import Table from "../../ui/Table";

interface IArtifactProps {
  name: string;
  artifact_type: string;
}

const ARTIFACTS_TABLE_HEADER = ["Name", "Type"];
const ARTIFACTS_TABLE_RATIO = [1, 4];

const ArtifactsTable: React.FC<{ artifacts: Array<IArtifactProps> }> = ({
  artifacts
}) => {
  const parsedArtifacts = artifacts.map(artifact => {
    return [artifact.name, artifact.artifact_type];
  });
  return (
    <div>
      <h2>Artifacts</h2>
      <Table
        content={parsedArtifacts}
        ratio={ARTIFACTS_TABLE_RATIO}
        header={ARTIFACTS_TABLE_HEADER}
      />
    </div>
  );
};

export default ArtifactsTable;
