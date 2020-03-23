import styled from "@emotion/styled";

export const TableContainer = styled.div({
  width: "100%",
  margin: "auto",
  backgroundColor: "#F5F8FA"
});

export const Row = styled.div<{ showBottomBorder?: boolean }>(props => ({
  display: "flex",
  flexWrap: "wrap",
  borderBottom: props.showBottomBorder ? "2px solid #CED9E0" : "none"
}));

export const TableHeader = styled(Row)({
  borderBottom: "2px solid #CED9E0",
  fontWeight: "bold"
});

export const Cell = styled.div<{ maxWidth?: number }>(props => ({
  color: "#202B33",
  textAlign: "left",
  padding: "20px",
  flex: 1,
  height: "auto",
  position: "relative",
  whiteSpace: "pre-wrap",
  overflowWrap: "break-word",
  maxWidth: props.maxWidth ? `${props.maxWidth}px` : null
}));
