import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const ENV_TABLE_RATIO = [1, 4];

const EnvTable: React.FC<{ env: { [key: string]: string } }> = ({ env }) => {
  const envKeys = Object.keys(env);

  const parsedEnv = envKeys.map((key) => {
    return { content: [key, env[key]] };
  });

  return (
    <Section>
      <h2>Environments</h2>
      <Table content={parsedEnv} ratio={ENV_TABLE_RATIO} />
    </Section>
  );
};

export default EnvTable;
