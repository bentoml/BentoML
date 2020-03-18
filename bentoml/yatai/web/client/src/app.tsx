import * as React from 'react';
import { Switch, Route } from 'react-router';
import { BrowserRouter, Link } from 'react-router-dom';
import {
  Navbar,
  NavbarGroup,
  NavbarHeading,
  NavbarDivider,
  Alignment,
  Button,
  Classes,
} from '@blueprintjs/core';
import { Home } from './pages/home';
import { DeploymentsList } from './pages/deployments_list';
import { DeploymentDetails } from './pages/deployment_details';
import { BentosList } from './pages/bentos_list';
import { BentoServicesList } from './pages/bento_services_list';
import { BentoServiceDetail } from './pages/bento_service_detail';

const HeaderComp = () => (
  <Navbar>
    <NavbarGroup align={Alignment.LEFT}>
      <NavbarHeading>BentoML</NavbarHeading>
      <NavbarDivider />
      <Link to='deployments'>
        <Button className={Classes.MINIMAL} icon="document" text="Deployments" />
      </Link>
      <Link to='bentos'>
        <Button className={Classes.MINIMAL} icon="document" text="BentoServices" />
      </Link>
    </NavbarGroup>
  </Navbar>
)

export const App = () => (
  <BrowserRouter>
    <div className='app'>
      <div>
        <HeaderComp />
      </div>
      <div>
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
    </div>
  </BrowserRouter>
);
