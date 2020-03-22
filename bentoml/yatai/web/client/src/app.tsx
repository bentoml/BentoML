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
import { Repository } from './pages/repository';
import { BentoServicesList } from './pages/bento_services_list';
import { BentoServiceDetail } from './pages/bento_service_detail';

const HeaderComp = () => (
  <Navbar>
    <NavbarGroup align={Alignment.LEFT}>
      <Link to='/'>
        <NavbarHeading>BentoML</NavbarHeading>
      </Link>
      <NavbarDivider />
      <Link to='/repository'>
        <Button className={Classes.MINIMAL} icon="document" text="Repository" />
      </Link>
      <Link to='/deployments'>
        <Button className={Classes.MINIMAL} icon="document" text="Deployments" />
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
          <Route path='/deployments/:namespace/:name' component={DeploymentDetails} />
          <Route path='/deployments' component={DeploymentsList} />

          <Route path='/repository/:name/:version' component={BentoServiceDetail} />
          <Route path='/repository/:name' component={BentoServicesList} />
          <Route path='/repository' component={Repository} />

          <Route path='/about' component={Home} / >
          <Route path='/config' component={Home} / >
          <Route exact path='/' component={Home} />
        </Switch>
      </div>
    </div>
  </BrowserRouter>
);
