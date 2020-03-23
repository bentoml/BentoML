import * as React from "react";
import { TableContainer, Row, Cell } from "../../ui/Table";

const EnvTable: React.FC<{ env: { [key: string]: string } }> = ({ env }) => {
  const envKeys = Object.keys(env);

  return (
    <div>
      <h2>Environments</h2>
      <TableContainer>
        {envKeys.map((envKey, i) => (
          <Row key={i} showBottomBorder={true}>
            <Cell maxWidth={200}>{envKey}</Cell>
            <Cell>{env[envKey]}</Cell>
          </Row>
        ))}
      </TableContainer>
    </div>
  );
};

export default EnvTable;
