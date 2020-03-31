import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

interface IArtifactProps {
  name: string;
  artifact_type: string;
}

const ARTIFACTS_TABLE_HEADER = ["Name", "Type"];
const ARTIFACTS_TABLE_RATIO = [1, 4];

const ArtifactsTable: React.FC<{ artifacts: Array<IArtifactProps> }> = ({
  artifacts
}) => {
  const artifactsTableContent = artifacts.map(artifact => {
    return {content:[artifact.name, artifact.artifact_type]};
  });
  return (
    <Section>
      <h2>Artifacts</h2>
      <Table
        content={artifactsTableContent}
        ratio={ARTIFACTS_TABLE_RATIO}
        header={ARTIFACTS_TABLE_HEADER}
      />
    </Section>
  );
};

export default ArtifactsTable;
