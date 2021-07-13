import * as React from "react";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const ENV_TABLE_RATIO = [1, 4];

const EnvTable: React.FC<{ env: { [key: string]: any } }> = ({ env }) => {
  const envKeys = Object.keys(env);

  const parsedEnv = envKeys.map((key) => {
    let env_value = env[key];
    if (key === "pip_packages") {
      env_value = env[key].join("\n");
    }
    return { content: [key, env_value] };
  });

  return (
    <Section>
      <h2>Environments</h2>
      <Table content={parsedEnv} ratio={ENV_TABLE_RATIO} />
    </Section>
  );
};

export default EnvTable;
