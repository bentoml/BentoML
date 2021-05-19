import * as lodash from "lodash";

const extractExpressionElements = (expression: string): Array<string> => {
  return ["", "", ""];
};

export const generateGrpcLabelSelectors = (selectors: Array<string>) => {
  let result: any = {
    matchLabels: {},
    matchExpressions: [],
  };

  lodash.forEach(selectors, (selector) => {
    const expression = extractExpressionElements(selector);

    switch (expression.length) {
      case 1:
        /*
        possible expressions: key, key=value, key!=value
        */
        const query = expression[0];
        if (query.includes("=")) {
          if ((query.match(/=/g) || []).length > 1) {
            throw new Error("");
          }
          const splitEqual = query.split("=");
          result.matchLabels[splitEqual[0]] = splitEqual[1];
        } else if (query.includes("!=")) {
          if ((query.match(/!=/g) || []).length > 1) {
            throw new Error("");
          }
          const splitNotEqual = query.split("!=");
          result.matchExpressions.push({
            operatorType: "NotIn",
            key: splitNotEqual[0],
            values: [splitNotEqual[1]],
          });
        } else {
          result.matchExpressions.push({
            operatorType: "Exists",
            key: query,
          });
        }
        break;
      case 2:
        /*
        possible expressions: key exists, key DoesNotExist
        */
        if (
          !["exists", "doesnotexist"].includes(lodash.toLower(expression[1]))
        ) {
          throw new Error("");
        }
        result.matchExpressions.push({
          operatorType:
            lodash.toLower(expression[1]) == "exists"
              ? "Exists"
              : "DoesNotExist",
          key: expression[0],
        });
        break;
      case 3:
        /*
        possible expression: key in (v1, v2), key notin (v1, v2)
        */
        if (!["in", "notin"].includes(lodash.toLower(expression[1]))) {
          throw new Error("");
        }
        const valueList = getValueList(expression[2]);
        result.matchExpressions.push({
          operatorType: lodash.toLower(expression[1]) == 'in' ? 'In' : 'NotIn',
          key: expression[0],
          values:valueList,
        });
        break;
      default:
        throw new Error("");
    }
  });
  return result;
};
