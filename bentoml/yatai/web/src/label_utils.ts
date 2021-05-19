import * as lodash from "lodash";

const extractExpressionElements = (expression: string) => {
  return [];
};

export const generateGrpcLabelSelectors = (selectors: [string]) => {
  return lodash.map(selectors, (selector) => {
    const expression = extractExpressionElements(selector);

    switch (expression.length) {
      case 1:
      case 2:
      case 3:
      default:
        throw new Error("");
    }
  });
};
