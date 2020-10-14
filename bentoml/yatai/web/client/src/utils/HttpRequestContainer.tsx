import * as React from "react";
import axios, { Method } from "axios";
import { Card, Spinner } from "@blueprintjs/core";

interface IHttpRequestContainerProps {
  url: string;
  data?: any;
  method?: Method;
  headers?: any;
  params?: any;
  children?: any;
}

const HttpRequestContainer: React.FC<IHttpRequestContainerProps> = (props) => {
  const [info, setInfo] = React.useState({
    data: undefined,
    error: undefined,
    isLoading: true,
  });

  React.useEffect(() => {
    fetch(undefined, props, setInfo);
  }, [props]);

  if (info.isLoading) {
    return <Spinner />;
  }
  if (info.error) {
    return <DisplayHttpError error={info.error} />;
  }
  return props.children({
    data: info.data,
  });
};

const DisplayHttpError = ({ error }) => {
  return (
    <Card>
      <h3>Encounter HTTP Request Error</h3>
      <div>{JSON.stringify(error)}</div>
    </Card>
  );
};

const instance = axios.create({
  baseURL: '/yatai'
});

const fetch = (options = {}, props, callback) => {
  const { url, data, method, headers, params } = Object.assign(
    {},
    props,
    options
  );

  return instance({ method, url, data, headers, params })
    .then((response) => {
      callback({ data: response.data, isLoading: false, error: false });
    })
    .catch((error) => {
      callback({ error: error.response, isLoading: false, data: undefined });
    });
};

export default HttpRequestContainer;
