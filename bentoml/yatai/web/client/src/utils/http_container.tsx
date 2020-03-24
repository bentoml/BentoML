import * as React from "react";
import axios, { Method } from "axios";


interface IHttpRequestContainer {
  url: string;
  data?: any;
  method?: Method;
  headers?: any;
  params?: any;
  children?: any;
}

export class HttpRequestContainer extends React.Component<IHttpRequestContainer> {
  state = {
    response: undefined,
    data: undefined,
    error: undefined,
    isLoading: true,
  };

  componentWillMount() {
    this.fetch();
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.params != this.props.params) {
      this.fetch();
    }
  }

  fetch = (options = {}) => {
    const { url, data, method, headers, params } = Object.assign(
      {},
      this.props,
      options
    );

    const updateProp = info => {
      this.setState(info);
    };

    return axios({ method, url, data, headers, params })
      .then(response => {
        updateProp({ data: response.data, isLoading: false});
      })
      .catch(error => {
        updateProp({ error: error.response, isLoading: false });
      });
  };

  render() {
    return this.props.children({
      isLoading: this.state.isLoading,
      data: this.state.data,
      error: this.state.error
    });
  }
}
