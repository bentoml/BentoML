import * as React from "react";
import { useLocation } from "react-router-dom";
import { capitalize } from "lodash";

import {
  Breadcrumbs as BlueprintBreadcrumbs,
  IBreadcrumbProps,
} from "@blueprintjs/core";
import { Section } from "../ui/Layout";


const Breadcrumbs: React.FC = (props) => {
  const [breadcrumbs, setBreadcrumbs] = React.useState<Array<IBreadcrumbProps>>(
    []
  );
  const location = useLocation();
  const {baseURL} = props;
  console.log(baseURL)
  const HOME_CRUMB: IBreadcrumbProps = {
    text: "Home",
    href: baseURL + '/'
  };

  React.useEffect(() => {
    const parsedBreadcrumbs = getBreadcrumbs(baseURL,location.pathname);
    setBreadcrumbs(parsedBreadcrumbs);
  }, [location]);

  breadcrumbs.length > 0 && breadcrumbs.unshift(HOME_CRUMB);
  if (breadcrumbs.length === 0) {
    return null;
  }
  return (
    <Section>
      <BlueprintBreadcrumbs items={breadcrumbs} />
    </Section>
  );
};

const getBreadcrumbs = (baseURL: string, pathname: string): Array<IBreadcrumbProps> => {
  const pathSnippets = pathname.split("/").filter((i) => i);

  return pathSnippets.map((name, index) => {
    const isLastOne = index === pathSnippets.length - 1;
    const url = `/${pathSnippets.slice(0, index + 1).join("/")}`;

    let formattedName = name;
    if (["repository", "deployments"].includes(name)) {
      formattedName = capitalize(name);
    }
    return isLastOne
      ? { text: formattedName }
      : { text: formattedName, href: baseURL + url };
  });
};

export default Breadcrumbs;
