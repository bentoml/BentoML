import * as React from 'react';
import axios from 'axios';
import * as moment from '../../node_modules/moment/moment';


interface IFetchContainerProps {
  url: string;
  data?: any;
  method?: string;
  headers?: any;
  params?: any;
  children?: any;
}

export class FetchContainer extends React.Component<IFetchContainerProps> {
  state = {
    response: undefined,
    data: undefined,
    error: undefined,
  };

  componentWillMount() {
    this.fetch();
  }

  fetch = (options = {}) => {
    const { url, data, method, headers, params } = Object.assign(
      {}, this.props, options
    );

    const updateProp = (info) => {
      this.setState(info);
    }

    return axios({method, url, data, headers, params})
      .then(response => {
        updateProp({data: response.data});
      })
      .catch(error => {
        updateProp({error: error.response});
      })
  }

  render () {
    return (
      this.props.children({
        data: this.state.data,
        error: this.state.error,
      })
    )
  }
};


export const displayTimeInFromNowFormat = (seconds: number) => {
  return moment.unix(seconds).fromNow();
};