import * as React from "react";
import styled from "@emotion/styled";
import { zip } from "lodash";

const TableContainer = styled.div({
  width: "100%",
  margin: "auto",
  backgroundColor: "#F5F8FA"
});

const Row = styled.div<{ showBottomBorder?: boolean }>(props => ({
  display: "flex",
  flexWrap: "wrap",
  borderBottom: props.showBottomBorder ? "2px solid #D8E1E8" : "none",
  ":hover": {
    backgroundColor: "#EBF1F5"
  }
}));

const TableHeader = styled(Row)({
  borderBottom: "2px solid #D8E1E8",
  fontWeight: 500,
  ":hover": {
    backgroundColor: "#F5F8FA"
  }
});

const Cell = styled.div<{
  flex?: number;
}>(props => ({
  color: "#202B33",
  textAlign: "left",
  padding: "20px",
  flex: props.flex ? props.flex : 1,
  height: "auto",
  position: "relative",
  whiteSpace: "pre-wrap",
  overflowWrap: "break-word",
  minWidth: "100px"
}));

interface ITableProps {
  content: Array<any>;
  header?: Array<string>;
  ratio?: Array<number>;
}

const Table: React.FC<ITableProps> = props => {
  const { content, header, ratio } = props;
  const finalHeader = ratio && header ? zip(header, ratio) : header;
  return (
    <TableContainer>
      {finalHeader && (
        <TableHeader>
          {finalHeader.map((h, i) => (
            <Cell key={i} flex={h[1]}>
              {h[0]}
            </Cell>
          ))}
        </TableHeader>
      )}
      {content.map((row, i) => {
        const r = zip(row, ratio);
        return (
          <Row key={i}>
            {r.map((cell, j) => (
              <Cell key={j} flex={cell[1]}>
                {cell[0]}
              </Cell>
            ))}
          </Row>
        );
      })}
    </TableContainer>
  );
};

export const TableNew: React.FC<ITableProps> = props => {
  const { content, header, ratio } = props;
  const finalHeader = ratio && header ? zip(header, ratio) : header;
  return (
    <TableContainer>
      {finalHeader && (
        <TableHeader>
          {finalHeader.map((h, i) => (
            <Cell key={i} flex={h[1]}>
              {h[0]}
            </Cell>
          ))}
        </TableHeader>
      )}
      {content.map((row, i) => {
        const r = zip(row.content, ratio);

        return (
          <Row key={i} onClick={() => handleClick(row.link)}>
            {r.map((cell, j) => (
              <Cell key={j} flex={cell[1]}>
                {cell[0]}
              </Cell>
            ))}
          </Row>
        );
      })}
    </TableContainer>
  );
};

const handleClick = (path: string) => {
  if (path) {
    window.location.href = path;
  }
};

export default Table;
