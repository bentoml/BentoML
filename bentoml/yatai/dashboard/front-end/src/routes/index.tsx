import * as React from 'react';
import { Switch, Route } from 'react-router-dom';

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

export const AppRoutes = () => (
  <Switch>
    <AppRoute exact path='/signin' component={HomePage} layout={MainLayout} />
    <AppRoute exact path='/settings' component={HomePage} layout={MainLayout} />
    <AppRoute exact path='/' component={HomePage} layout={MainLayout} />

    <AppRoute path='/deployments/:name' component={HomePage} layout={MainLayout} />
    <AppRoute path='/deployments' component={HomePage} layout={MainLayout} />

    <AppRoute path='/bentos/:name/:version' component={HomePage} layout={MainLayout} />
    <AppRoute path='/bentos/:name' component={HomePage} layout={MainLayout} />
    <AppRoute path='/bentos' component={HomePage} layout={MainLayout} />
  </Switch>
)
