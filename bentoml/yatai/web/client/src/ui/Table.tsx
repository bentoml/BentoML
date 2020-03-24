import * as React from "react";
import styled from "@emotion/styled";
import { zip } from "lodash";

export const TableContainer = styled.div({
  width: "100%",
  margin: "auto",
  backgroundColor: "#F5F8FA"
});

export const Row = styled.div<{ showBottomBorder?: boolean }>(props => ({
  display: "flex",
  flexWrap: "wrap",
  borderBottom: props.showBottomBorder ? "2px solid #D8E1E8" : "none"
}));

export const TableHeader = styled(Row)({
  borderBottom: "2px solid #D8E1E8",
  fontWeight: 500
});

export const Cell = styled.div<{
  maxWidth?: number;
  color?: string;
  flex?: number;
}>(props => ({
  color: props.color ? props.color : "#202B33",
  textAlign: "left",
  padding: "20px",
  flex: props.flex ? props.flex : 1,
  height: "auto",
  position: "relative",
  whiteSpace: "pre-wrap",
  overflowWrap: "break-word",
  maxWidth: props.maxWidth ? `${props.maxWidth}px` : null
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

export default Table;
