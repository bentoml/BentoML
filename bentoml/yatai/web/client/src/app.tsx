import * as React from 'react';
import axios from 'axios';
import { Switch, Route } from 'react-router';
import { BrowserRouter } from 'react-router-dom';
import { Home } from './pages/home';
import { DeploymentsList } from './pages/deployments_list';
import { DeploymentDetails } from './pages/deployment_details';
import { BentosList } from './pages/bentos_list';
import { BentoServicesList } from './pages/bento_services_list';
import { BentoServiceDetail } from './pages/bento_service_detail';

interface IFetchContainerProps {
  url: string;
  data?: any;
  method?: string;
  headers?: any;
  params?: any;
  children?: any;
}

class FetchContainer extends React.Component<IFetchContainerProps> {
  state = {
    response: undefined,
    data: undefined,
    error: undefined,
  };

  componentDidMount() {
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
}
// <FetchContainer url='/api/listBento' method='get'>
//   {({ data, error }) => (
//     <div>
//       <pre>{JSON.stringify(data, null , 2)}</pre>
//     </div>
//   )}
// </FetchContainer>

export const App = () => (
  <BrowserRouter>
    <div className='app'>
      <Switch>
        <Route path='/deployments/:name' component={DeploymentDetails} />
        <Route path='/deployments' component={DeploymentsList} />

        <Route path='/bentos/:name/:version' component={BentoServiceDetail} />
        <Route path='/bentos/:name' component={BentoServicesList} />
        <Route path='/bentos' component={BentosList} />

        <Route path='/about' component={Home} / >
        <Route path='/config' component={Home} / >
        <Route exact path='/' component={Home} />
      </Switch>
    </div>
  </BrowserRouter>
);
