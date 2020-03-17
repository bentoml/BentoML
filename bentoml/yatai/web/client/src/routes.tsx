import * as React from 'react';
import { Switch, Route } from 'react-router-dom';
import { Home } from './pages/home';
import { DeploymentsList } from './pages/deployments_list';
import { DeploymentDetails } from './pages/deployment_details';
import { BentoServiceDetail } from './pages/bento_service_detail';
import { BentosList } from './pages/bentos_list';
import { BentoServicesList } from './pages/bento_services_list';


export const AppRoute = () => (
  <Switch>
    <Route path='/'><Home /></Route>
    <Route path='/about'><Home /></Route>
    <Route path='/config'><Home /></Route>
    <Route path='/deployments'><DeploymentsList /></Route>
    <Route path='/deployments/:name'><DeploymentDetails /></Route>
    <Route path='/bentos'><BentosList /></Route>
    <Route path='/bentos/:name'><BentoServicesList /></Route>
    <Route path='/bentos/:name/:version'><BentoServiceDetail /></Route>
  </Switch>
);