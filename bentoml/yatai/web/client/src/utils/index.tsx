import * as moment from "../../node_modules/moment/moment";
import * as qs from "qs";

export const displayTimeInFromNowFormat = (seconds: number, displayAgoString: boolean=false) => {
  return moment.unix(seconds).fromNow(!displayAgoString);
};

export const getQueryObject = queryString => {
  return qs.parse(queryString, { ignoreQueryPrefix: true });
};
