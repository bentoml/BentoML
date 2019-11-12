import * as React from 'react';
import { Switch, Route } from 'react-router-dom';

import { HomePage } from '../components/home-page';
import { prependOnceListener } from 'cluster';

interface RouteArgs {
  component: React.SFC<any> | React.ComponentClass<any>;
  layout: React.SFC<any>;
  exact?: boolean;
  path?: string;
}

const AppRoute = ({component: Component, layout: Layout, ...rest}: RouteArgs) => (
  <Route {...rest} render={props => (
    <Layout>
      <Component {...props} />
    </Layout>
  )} />
);

const MainLayout = (props: {children: any}) => (
  <div>
    {props.children}
  </div>
);

export const AppRoutes = () => (
  <Switch>
    <AppRoute exact path='/' component={HomePage} layout={MainLayout} />

    <AppRoute path='/deployments/:namespace/:name' component={HomePage} layout={MainLayout} />
    <AppRoute path='/deployments/:namespace' component={HomePage} layout={MainLayout} />
    <AppRoute path='/deployments' component={HomePage} layout={MainLayout} />

    <AppRoute path='/bentos/:name/:version' component={HomePage} layout={MainLayout} />
    <AppRoute path='/bentos/:name' component={HomePage} layout={MainLayout} />
    <AppRoute path='/bentos' component={HomePage} layout={MainLayout} />
  </Switch>
)
